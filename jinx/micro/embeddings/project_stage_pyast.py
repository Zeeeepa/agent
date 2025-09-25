from __future__ import annotations

import ast
import os
import re
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files


def _extract_call_names(q: str) -> List[str]:
    q = (q or "").strip()
    if not q:
        return []
    names: List[str] = []
    # Find identifiers followed by '('
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", q):
        name = m.group(1)
        if name and name not in names:
            names.append(name)
    return names


def stage_pyast_hits(query: str, k: int, *, max_time_ms: int | None = 220) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage -2: Python AST-driven match for call names from the query.

    Targets Python files only and searches for ast.Call occurrences of candidate
    function names extracted from the query. Returns list of (score, file_rel, obj).
    """
    t0 = time.perf_counter()
    cand = _extract_call_names(query)
    if not cand:
        return []

    hits: List[Tuple[float, str, Dict[str, Any]]] = []
    seen: set[str] = set()

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    for abs_p, rel_p in iter_candidate_files(
        ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES
    ):
        if time_up():
            break
        if rel_p in seen:
            continue
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

        call_lines: List[int] = []

        class V(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:  # type: ignore[name-defined]
                fn = node.func
                name: str | None = None
                if isinstance(fn, ast.Name):
                    name = fn.id
                elif isinstance(fn, ast.Attribute):
                    name = fn.attr
                if name and name in cand:
                    ln = getattr(node, "lineno", None)
                    if isinstance(ln, int):
                        call_lines.append(ln)
                self.generic_visit(node)

        V().visit(tree)
        if not call_lines:
            continue
        seen.add(rel_p)
        lines = src.splitlines()
        # Choose first occurrence for the preview window
        ln = max(1, call_lines[0])
        a = max(1, ln - 12)
        b = min(len(lines), ln + 12)
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
        # AST hits are very precise, score slightly above phrase hits
        hits.append((0.995, rel_p, obj))
        if len(hits) >= k:
            break
    return hits


__all__ = ["stage_pyast_hits"]
