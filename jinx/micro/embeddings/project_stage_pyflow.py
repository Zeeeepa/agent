from __future__ import annotations

import os
import ast
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks


def _tokens_from_query(q: str) -> List[str]:
    import re
    raw: List[str] = []
    for m in re.finditer(r"[A-Za-z_][\w\.]{1,}", q or ""):
        s = (m.group(0) or "").strip()
        if not s:
            continue
        raw.append(s.lower())
    # Expand dotted and underscore splits to improve recall
    exp: List[str] = []
    for t in raw:
        exp.append(t)
        if "." in t:
            parts = [p for p in t.split(".") if p]
            exp.extend([p for p in parts if len(p) >= 3])
        if "_" in t:
            parts = [p for p in t.split("_") if p]
            exp.extend([p for p in parts if len(p) >= 3])
    # Dedupe preserving order
    out: List[str] = []
    seen: set[str] = set()
    for t in exp:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:24]


def _attr_chain(n: ast.AST) -> List[str]:
    # Collect attribute chain parts: db.get -> ["db", "get"]
    out: List[str] = []
    cur = n
    while isinstance(cur, ast.Attribute):
        try:
            out.append(str(cur.attr))
        except Exception:
            break
        cur = cur.value  # type: ignore[assignment]
    if isinstance(cur, ast.Name):
        out.append(cur.id)
    out.reverse()
    return out


def _name_of_func(n: ast.AST) -> List[str]:
    # Return possible names for the call target
    if isinstance(n, ast.Name):
        return [n.id]
    if isinstance(n, ast.Attribute):
        chain = _attr_chain(n)
        if chain:
            # include full path and the last element
            names = [".".join(chain), chain[-1]]
            # also include head if length > 1
            if len(chain) > 1:
                names.append(chain[0])
            return list(dict.fromkeys([s.lower() for s in names]))
    return []


def _names_from_arg(a: ast.AST) -> List[str]:
    # Extract plain names from args: rel_path, self.rel_path -> rel_path, self, rel_path
    out: List[str] = []
    if isinstance(a, ast.Name):
        out.append(a.id)
    elif isinstance(a, ast.Attribute):
        out.extend(_attr_chain(a))
    elif isinstance(a, ast.Constant) and isinstance(a.value, str):
        # allow matching string literal arguments
        out.append(str(a.value))
    return [s.lower() for s in out if s]


def _flatten_targets(t: ast.AST) -> List[str]:
    out: List[str] = []
    def _walk(x: ast.AST) -> None:
        if isinstance(x, ast.Name):
            out.append(x.id)
        elif isinstance(x, (ast.Tuple, ast.List)):
            for e in x.elts:
                if isinstance(e, ast.AST):
                    _walk(e)
    _walk(t)
    return [s.lower() for s in out if s]


def _names_from_genexp(g: ast.GeneratorExp) -> List[str]:
    names: List[str] = []
    # element
    try:
        elt = g.elt
        if isinstance(elt, ast.Call):
            names.extend(_name_of_func(elt.func))
            for a in list(elt.args) + [kw.value for kw in elt.keywords]:
                names.extend(_names_from_arg(a))
        elif isinstance(elt, ast.Attribute):
            names.extend(_attr_chain(elt))
        elif isinstance(elt, ast.Name):
            names.append(elt.id)
    except Exception:
        pass
    # generators
    try:
        for comp in g.generators:
            try:
                names.extend(_flatten_targets(comp.target))
            except Exception:
                pass
            try:
                if isinstance(comp.iter, ast.Call):
                    names.extend(_name_of_func(comp.iter.func))
                elif isinstance(comp.iter, ast.Attribute):
                    names.extend(_attr_chain(comp.iter))
                elif isinstance(comp.iter, ast.Name):
                    names.append(comp.iter.id)
            except Exception:
                pass
    except Exception:
        pass
    # dedupe
    out: List[str] = []
    seen: set[str] = set()
    for n in [s.lower() for s in names if s]:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def stage_pyflow_hits(query: str, k: int, *, max_time_ms: int | None = 220) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: Python control/flow pattern search (e.g., 'return db.get(rel_path)').

    Looks for Return(Call(...)) matching tokens from the query: function names/attributes and argument names.
    """
    q = (query or "").strip()
    if not q:
        return []
    q_toks = _tokens_from_query(q)
    if not q_toks:
        return []

    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    # Helper to process one file
    def _process_file(abs_p: str, rel_p: str) -> bool:
        if time_up():
            return True
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
        except Exception:
            return False
        if not src:
            return False
        # Cheap prefilter: require at least one query token present in file text
        try:
            low = src.lower()
            if not any(t for t in q_toks if t and t in low):
                return False
        except Exception:
            pass
        try:
            tree = ast.parse(src)
        except Exception:
            return False
        lines = src.splitlines()

        best_local: List[Tuple[float, int, int]] = []

        class V(ast.NodeVisitor):
            def visit_Return(self, node: ast.Return) -> None:  # type: ignore[name-defined]
                if time_up():
                    return
                val = node.value
                if not isinstance(val, ast.Call):
                    return
                fn_names = _name_of_func(val.func)
                arg_names: List[str] = []
                try:
                    for a in list(val.args) + [kw.value for kw in val.keywords]:
                        arg_names.extend(_names_from_arg(a))
                except Exception:
                    pass
                # scoring
                score = 0.0
                qset = set(q_toks)
                # function target
                for nm in fn_names:
                    if nm in qset:
                        score += 1.0
                # special: split dotted full path and check parts
                for nm in fn_names:
                    if "." in nm:
                        for p in nm.split("."):
                            if p in qset:
                                score += 0.5
                # arguments
                for an in arg_names:
                    if an in qset:
                        score += 0.75
                # direct 'return' token grants small bonus
                if "return" in qset:
                    score += 0.5
                # heuristic threshold
                if score >= 1.5:
                    ln = getattr(node, "lineno", 0) or 0
                    if ln > 0:
                        a = max(1, ln - 12)
                        b = min(len(lines), ln + 12)
                        best_local.append((score, a, b))
                self.generic_visit(node)

        V().visit(tree)
        # Extend visitor with Assign patterns
        class V2(ast.NodeVisitor):
            def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[name-defined]
                if time_up():
                    return
                val = node.value
                score = 0.0
                names: List[str] = []
                if isinstance(val, ast.Call):
                    # already covered by libcst Assign(Call), but include here for completeness
                    names.extend(_name_of_func(val.func))
                    for a in list(val.args) + [kw.value for kw in val.keywords]:
                        names.extend(_names_from_arg(a))
                elif isinstance(val, ast.GeneratorExp):
                    names.extend(_names_from_genexp(val))
                elif isinstance(val, (ast.ListComp, ast.SetComp)):
                    try:
                        # Treat like generator: visit elt and generators
                        fake_gen = ast.GeneratorExp(val.elt, val.generators)  # type: ignore[arg-type]
                        names.extend(_names_from_genexp(fake_gen))
                    except Exception:
                        pass
                elif isinstance(val, ast.DictComp):
                    try:
                        # Include keys/values names conservatively
                        if isinstance(val.key, ast.AST):
                            names.extend(_names_from_arg(val.key))
                        if isinstance(val.value, ast.AST):
                            names.extend(_names_from_arg(val.value))
                        for comp in val.generators:
                            names.extend(_flatten_targets(comp.target))
                    except Exception:
                        pass
                if not names:
                    return
                qset = set(q_toks)
                for nm in names:
                    if nm in qset:
                        score += 0.7
                # include assignment targets as strong hints
                try:
                    for tgt in node.targets:
                        for tname in _flatten_targets(tgt):
                            if tname in qset:
                                score += 0.8
                except Exception:
                    pass
                if score >= 1.5:
                    ln = getattr(node, "lineno", 0) or 0
                    if ln > 0:
                        a = max(1, ln - 12)
                        b = min(len(lines), ln + 12)
                        best_local.append((score, a, b))
                self.generic_visit(node)

        V2().visit(tree)
        if best_local:
            # Take the top match for this file
            best_local.sort(key=lambda t: t[0], reverse=True)
            _, a, b = best_local[0]
            preview = "\n".join(lines[a - 1 : b]).strip()
            obj = {
                "embedding": [],
                "meta": {
                    "file_rel": rel_p,
                    "text_preview": preview,
                    "line_start": a,
                    "line_end": b,
                },
            }
            hits.append((0.9935, rel_p, obj))
            return len(hits) >= k
        return False

    # Scan embeddings-known files first
    try:
        rel_files: List[str] = []
        seen_f: set[str] = set()
        for fr, obj in iter_project_chunks():
            rel = fr or str((obj.get("meta") or {}).get("file_rel") or "")
            if rel and rel not in seen_f:
                seen_f.add(rel)
                rel_files.append(rel)
        for rel_p in rel_files:
            abs_p = os.path.join(ROOT, rel_p)
            if _process_file(abs_p, rel_p):
                return hits[:k]
    except Exception:
        pass

    # Fallback: full project walk
    for abs_p, rel_p in iter_candidate_files(
        ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES
    ):
        if _process_file(abs_p, rel_p):
            return hits[:k]

    return hits


__all__ = ["stage_pyflow_hits"]
