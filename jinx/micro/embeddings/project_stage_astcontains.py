from __future__ import annotations

import ast
import os
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks

# Simple token check to gate pattern activation
_DEF_MIN_LEN = 3


def _has_tokens(q: str, tokens: List[str]) -> bool:
    s = (q or "").lower()
    return all(t in s for t in tokens)


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _is_ast_type_expr(n: ast.AST) -> bool:
    # True for expressions like ast.Name, ast.Call, ast.Attribute, etc.
    # Accept nested attribute chains ast.foo.bar as well.
    def _match_attr(x: ast.AST) -> bool:
        if isinstance(x, ast.Attribute):
            # Recurse on base and allow any attribute name
            return _match_attr(x.value) or (isinstance(x.value, ast.Name) and x.value.id == "ast")
        if isinstance(x, ast.Name):
            return x.id == "ast"
        return False
    return _match_attr(n)


def _window(lines: List[str], s: int, e: int, around: int = 12) -> tuple[int, int, str]:
    a = max(1, s - around)
    b = min(len(lines), max(s, e) + around)
    snip = "\n".join(lines[a-1:b]).strip()
    return a, b, snip


def stage_astcontains_hits(query: str, k: int, *, max_time_ms: int | None = 200) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Python AST structural contains stage.

    Currently supports detection of patterns like:
    - isinstance(<expr>, ast.<Type>)

    This is gated by query tokens to avoid overhead on unrelated queries.
    """
    q = (query or "").strip()
    if not q or len(q) < _DEF_MIN_LEN:
        return []
    # Cheap gate: only run if both tokens are present in the query
    ql = q.lower()
    need = ("isinstance" in ql) and ("ast." in ql)
    if not need:
        return []
    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    def process(abs_p: str, rel_p: str) -> bool:
        if time_up():
            return True
        src = _read_text(abs_p)
        if not src:
            return False
        try:
            tree = ast.parse(src)
        except Exception:
            return False
        lines = src.splitlines()
        for node in ast.walk(tree):
            if time_up():
                return True
            try:
                if isinstance(node, ast.Call):
                    # Check func is 'isinstance' (Name or Attribute.*isinstance)
                    fn = node.func
                    ok_fn = False
                    if isinstance(fn, ast.Name):
                        ok_fn = (fn.id == "isinstance")
                    elif isinstance(fn, ast.Attribute):
                        ok_fn = (getattr(fn, "attr", "") == "isinstance")
                    if not ok_fn:
                        continue
                    # Need second arg to be ast.<Type>
                    args = list(getattr(node, "args", []) or [])
                    if len(args) < 2:
                        continue
                    if not _is_ast_type_expr(args[1]):
                        continue
                    s = int(getattr(node, "lineno", 0) or 0)
                    e = int(getattr(node, "end_lineno", 0) or s)
                    if s <= 0:
                        continue
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
                    hits.append((0.998, rel_p, obj))
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
            if rel and rel.endswith(".py") and rel not in seen:
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


__all__ = ["stage_astcontains_hits"]
