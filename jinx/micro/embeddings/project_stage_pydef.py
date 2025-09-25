from __future__ import annotations

import ast
import time
from typing import Any, Dict, List, Tuple, Sequence

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files


def _ltxt(s: str) -> str:
    return (s or "").strip().lower()


def _tokset(q: str, limit: int = 24) -> List[str]:
    import re
    toks: List[str] = []
    for m in re.finditer(r"[A-Za-z_][\w\.]{1,}", q or ""):
        t = _ltxt(m.group(0))
        if t and t not in toks:
            toks.append(t)
    return toks[:limit]


def _name_tokens(name: str) -> List[str]:
    # split by underscores and CamelCase
    import re
    parts: List[str] = []
    s = _ltxt(name)
    if not s:
        return parts
    parts.extend([p for p in s.split("_") if p])
    parts.extend([p for p in re.split(r"(?<!^)(?=[A-Z])", name) if p])
    out: List[str] = []
    seen: set[str] = set()
    for p in parts:
        pl = _ltxt(p)
        if pl and pl not in seen:
            seen.add(pl)
            out.append(pl)
    return out


def _ann_to_text(a: ast.AST | None) -> str:
    if a is None:
        return ""
    try:
        import astunparse  # type: ignore
        return _ltxt(astunparse.unparse(a))
    except Exception:
        # fallback rough repr
        try:
            return _ltxt(getattr(a, "id", "") or getattr(getattr(a, "attr", None), "value", ""))
        except Exception:
            return ""


def stage_pydef_hits(query: str, k: int, *, max_time_ms: int | None = 180) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: Python definitions (def/class) by name/signature tokens.

    Matches tokens against definition names, parameters, annotations, and decorators.
    """
    q = (query or "").strip()
    if not q:
        return []
    toks = _tokset(q)
    if not toks:
        return []
    tset = set(toks)

    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    for abs_p, rel_p in iter_candidate_files(
        ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES
    ):
        if time_up():
            break
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
        except Exception:
            continue
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        lines = src.splitlines()

        best_local: Tuple[float, int, int] | None = None

        def _score_name(nm: str) -> float:
            sc = 0.0
            nm_l = _ltxt(nm)
            if nm_l in tset:
                sc += 1.0
            for p in _name_tokens(nm):
                if p in tset:
                    sc += 0.5
            return sc

        def _score_decorators(decs: Sequence[ast.AST]) -> float:
            sc = 0.0
            for d in decs:
                # decorator could be Name or Attribute
                try:
                    if isinstance(d, ast.Name):
                        if _ltxt(d.id) in tset:
                            sc += 0.4
                    elif isinstance(d, ast.Attribute):
                        nm = _ltxt(getattr(d, "attr", ""))
                        if nm in tset:
                            sc += 0.4
                except Exception:
                    pass
            return sc

        def _score_params(args: ast.arguments) -> float:
            sc = 0.0
            params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
            if getattr(args, "vararg", None):
                params.append(args.vararg)  # type: ignore[arg-type]
            if getattr(args, "kwarg", None):
                params.append(args.kwarg)  # type: ignore[arg-type]
            for a in params:
                try:
                    nm = _ltxt(getattr(a, "arg", ""))
                    if nm in tset:
                        sc += 0.4
                    ann = _ann_to_text(getattr(a, "annotation", None))
                    if ann and any(t in ann for t in tset):
                        sc += 0.3
                except Exception:
                    pass
            return sc

        class V(ast.NodeVisitor):
            def _visit_def(self, node: ast.AST, name: str, decs: Sequence[ast.AST] | None, args: ast.arguments | None, returns: ast.AST | None, end_lineno: int | None) -> None:
                nonlocal best_local
                if time_up():
                    return
                sc = 0.0
                sc += _score_name(name)
                if decs:
                    sc += _score_decorators(decs)
                if args:
                    sc += _score_params(args)
                if returns is not None:
                    ann = _ann_to_text(returns)
                    if ann and any(t in ann for t in tset):
                        sc += 0.3
                if sc <= 0.0:
                    return
                ln = getattr(node, "lineno", 0) or 0
                if ln <= 0:
                    return
                a = max(1, ln - 12)
                b = min(len(lines), (end_lineno or ln) + 12)
                cur = (sc, a, b)
                if best_local is None or cur[0] > best_local[0]:
                    best_local = cur

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[name-defined]
                self._visit_def(node, node.name, node.decorator_list, node.args, node.returns, getattr(node, "end_lineno", None))
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[name-defined]
                self._visit_def(node, node.name, node.decorator_list, node.args, node.returns, getattr(node, "end_lineno", None))
                self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[name-defined]
                bases = []
                try:
                    for b in node.bases:
                        bases.append(_ann_to_text(b))
                except Exception:
                    pass
                sc_bases = 0.0
                for s in bases:
                    if any(t in s for t in tset):
                        sc_bases += 0.3
                # Use function scorer with name weight and base score
                nonlocal best_local
                name_sc = _score_name(node.name)
                sc = name_sc + sc_bases
                if sc > 0.0:
                    ln = getattr(node, "lineno", 0) or 0
                    if ln > 0:
                        a = max(1, ln - 12)
                        b = min(len(lines), int(getattr(node, "end_lineno", ln)) + 12)
                        cur = (sc, a, b)
                        if best_local is None or cur[0] > best_local[0]:
                            best_local = cur
                self.generic_visit(node)

        V().visit(tree)
        if best_local is not None:
            sc, a, b = best_local
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
            hits.append((0.9931, rel_p, obj))
            if len(hits) >= k:
                break

    return hits


__all__ = ["stage_pydef_hits"]
