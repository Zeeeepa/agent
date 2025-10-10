from __future__ import annotations

import ast
import os
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks


def _time_is_up(t0: float, limit_ms: int | None) -> bool:
    return limit_ms is not None and (time.perf_counter() - t0) * 1000.0 > limit_ms


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _node_skeleton(n: ast.AST, *, include_names: bool) -> Any:
    """Return a canonical, hashable skeleton of an AST node.

    If include_names is True, keep identifier/attribute names; otherwise, drop them.
    Constants are reduced to their type names.
    """
    t = type(n).__name__
    # Handle leaves and simple nodes
    if isinstance(n, ast.Name):
        return (t, n.id if include_names else "_")
    if isinstance(n, ast.Attribute):
        base = _node_skeleton(n.value, include_names=include_names)
        return (t, base, n.attr if include_names else "_")
    if isinstance(n, ast.Constant):
        ct = type(n.value).__name__
        return (t, ct)
    if isinstance(n, ast.alias):
        return (t, (n.name if include_names else "_"))
    # Recurse over fields
    items: List[Any] = [t]
    for field in ast.iter_fields(n):
        key, val = field
        if isinstance(val, ast.AST):
            items.append((key, _node_skeleton(val, include_names=include_names)))
        elif isinstance(val, list):
            items.append((key, tuple(_node_skeleton(x, include_names=include_names) if isinstance(x, ast.AST) else x for x in val)))
        else:
            # Keep operator types by name; drop numeric values
            if isinstance(val, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
                items.append((key, type(val).__name__))
            else:
                # keep small enums like ctx
                items.append((key, type(val).__name__))
    return tuple(items)


def _informative(node: ast.AST) -> ast.AST:
    """Pick a more informative subnode (e.g., Call) from a larger node when possible."""
    try:
        # Unwrap BoolOp by preferring non-constant side
        if isinstance(node, ast.BoolOp):
            vals = list(getattr(node, "values", []) or [])
            for v in vals:
                if not isinstance(v, ast.Constant):
                    return _informative(v)
            return vals[0] if vals else node
        # Prefer the call itself if present under unary/bool contexts
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.AST):
            return _informative(node.operand)
        if isinstance(node, ast.IfExp):
            return _informative(node.test)
        # If the node is a Call, it's already informative
        if isinstance(node, ast.Call):
            return node
        # For Compare, pick the left (often the expression of interest)
        if isinstance(node, ast.Compare):
            return _informative(node.left)
    except Exception:
        pass
    return node


def _extract_query_node(query: str) -> ast.AST | None:
    q = (query or "").strip()
    if not q:
        return None
    # Try as a module (statement)
    try:
        mod = ast.parse(q)
        if getattr(mod, "body", None):
            return mod.body[0]
    except Exception:
        pass
    # Try as expression
    try:
        expr = ast.parse(q, mode="eval")
        return _informative(expr.body)  # type: ignore[attr-defined]
    except Exception:
        pass
    # Salvage common boolean fragments like: 'and isinstance(x, T)' or 'call() and'
    try:
        s = q
        # Normalize C-like operators
        s = s.replace("&&", " and ").replace("||", " or ")
        s_stripped = s.strip()
        def _try_eval(src: str) -> ast.AST | None:
            try:
                return ast.parse(src, mode="eval").body  # type: ignore[attr-defined]
            except Exception:
                return None
        # Leading boolean operator
        if s_stripped.startswith("and ") or s_stripped.startswith("or "):
            cand = f"True {s_stripped}"
            node = _try_eval(cand)
            if node is not None:
                return _informative(node)
        # Trailing boolean operator
        if s_stripped.endswith(" and") or s_stripped.endswith(" or"):
            cand = f"{s_stripped} True"
            node = _try_eval(cand)
            if node is not None:
                return node
        # If-expression context
        try:
            m = ast.parse(f"if {s_stripped}:\n    pass")
            if getattr(m, "body", None):
                test = getattr(m.body[0], "test", None)
                if isinstance(test, ast.AST):
                    return _informative(test)
        except Exception:
            pass
    except Exception:
        pass
    return None


def _window(lines: List[str], s: int, e: int, around: int = 12) -> tuple[int, int, str]:
    a = max(1, s - around)
    b = min(len(lines), max(s, e) + around)
    snip = "\n".join(lines[a-1:b]).strip()
    return a, b, snip


def stage_astmatch_hits(query: str, k: int, *, max_time_ms: int | None = 220) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Match the query's AST shape against project Python files.

    Two-pass matching:
    1) strict (includes identifier/attribute names)
    2) loose (ignores names)

    Stops after the first k hits. Applies time budget aggressively.
    """
    qnode = _extract_query_node(query)
    if qnode is None:
        return []
    t0 = time.perf_counter()

    # Precompute query skeletons
    sk_strict = _node_skeleton(qnode, include_names=True)
    sk_loose = _node_skeleton(qnode, include_names=False)

    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def process(abs_p: str, rel_p: str) -> bool:
        if _time_is_up(t0, max_time_ms):
            return True
        src = _read_text(abs_p)
        if not src:
            return False
        try:
            tree = ast.parse(src)
        except Exception:
            return False
        # Build index of all nodes
        for node in ast.walk(tree):
            if _time_is_up(t0, max_time_ms):
                return True
            try:
                sk = _node_skeleton(node, include_names=True)
                ok = (sk == sk_strict)
                if not ok:
                    sk2 = _node_skeleton(node, include_names=False)
                    ok = (sk2 == sk_loose)
                if ok:
                    s = int(getattr(node, "lineno", 0) or 0)
                    e = int(getattr(node, "end_lineno", 0) or s)
                    if s <= 0:
                        continue
                    lines = src.splitlines()
                    a, b, snip = _window(lines, s, e)
                    obj = {
                        "embedding": [],
                        "meta": {
                            "file_rel": rel_p,
                            "text_preview": snip or "\n".join(lines[max(0, s-1):min(len(lines), e)]).strip(),
                            "line_start": a,
                            "line_end": b,
                        },
                    }
                    # Higher score for strict match
                    score = 0.999 if sk == sk_strict else 0.992
                    hits.append((score, rel_p, obj))
                    if len(hits) >= k:
                        return True
            except Exception:
                continue
        return False

    # Prefer embeddings-known files first
    try:
        seen: set[str] = set()
        rel_files: List[str] = []
        for fr, obj in iter_project_chunks():
            rel = fr or str((obj.get("meta") or {}).get("file_rel") or "")
            if rel and rel not in seen:
                seen.add(rel)
                rel_files.append(rel)
        for rel in rel_files:
            ap = os.path.join(ROOT, rel)
            if process(ap, rel):
                return hits[:k]
    except Exception:
        pass

    # Fallback: full walk limited to .py files
    for ap, rel in iter_candidate_files(ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if process(ap, rel):
            return hits[:k]

    return hits[:k]


__all__ = ["stage_astmatch_hits"]
