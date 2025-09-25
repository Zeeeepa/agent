from __future__ import annotations

import ast
import os
from typing import List, Tuple

from .project_config import ROOT, INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks


def extract_callees_from_scope(code: str, max_items: int = 10) -> List[str]:
    """Extract direct callee names from a Python scope body.

    Returns unique names in order of first appearance, truncated to max_items.
    For attribute calls like obj.method(), returns 'method'.
    """
    names: List[str] = []
    seen: set[str] = set()
    try:
        tree = ast.parse(code or "")
    except Exception:
        return []
    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # type: ignore[name-defined]
            nm = None
            try:
                func = node.func
                if isinstance(func, ast.Name):
                    nm = func.id
                elif isinstance(func, ast.Attribute):
                    nm = func.attr
            except Exception:
                nm = None
            if nm:
                lo = nm.lower()
                if lo not in seen:
                    seen.add(lo)
                    names.append(nm)
            self.generic_visit(node)
    V().visit(tree)
    return names[:max_items]


def _find_def_in_file(abs_path: str, rel_path: str, name: str) -> Tuple[int, int] | None:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
    except Exception:
        return None
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    s = e = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nm = getattr(node, "name", "")
            if nm == name:
                try:
                    s = int(getattr(node, "lineno", 0) or 0)
                    e = int(getattr(node, "end_lineno", 0) or 0) or s
                except Exception:
                    s, e = 0, 0
                if s > 0:
                    return (s, e)
    return None


def find_def_scope_in_project(symbol: str, prefer_rel: str | None = None, *, limit: int = 1) -> List[Tuple[str, int, int]]:
    """Find definition scopes for a Python symbol across the project.

    Returns list of (file_rel, line_start, line_end), preferring the provided file first.
    """
    nm = (symbol or "").strip()
    if not nm:
        return []
    got: List[Tuple[str, int, int]] = []
    # Prefer same file
    if prefer_rel:
        abs_p = os.path.join(ROOT, prefer_rel)
        de = _find_def_in_file(abs_p, prefer_rel, nm)
        if de:
            got.append((prefer_rel, de[0], de[1]))
            if len(got) >= limit:
                return got
    # Pass 1: embeddings-known files
    seen: set[str] = set()
    for fr, _obj in iter_project_chunks():
        if fr and fr != prefer_rel and fr not in seen:
            seen.add(fr)
            abs_p = os.path.join(ROOT, fr)
            de = _find_def_in_file(abs_p, fr, nm)
            if de:
                got.append((fr, de[0], de[1]))
                if len(got) >= limit:
                    return got
    # Pass 2: general project walk
    for abs_p, rel_p in iter_candidate_files(ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if rel_p and rel_p != prefer_rel and rel_p not in seen:
            seen.add(rel_p)
            de = _find_def_in_file(abs_p, rel_p, nm)
            if de:
                got.append((rel_p, de[0], de[1]))
                if len(got) >= limit:
                    break
    return got


__all__ = ["extract_callees_from_scope", "find_def_scope_in_project"]
