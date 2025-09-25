from __future__ import annotations

import ast
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files


def _norm(s: str) -> str:
    s2 = (s or "").lower()
    return " ".join(s2.split())


def _query_tokens(q: str, limit: int = 24) -> List[str]:
    import re
    toks: List[str] = []
    for m in re.finditer(r"[A-Za-z_][\w\.]{1,}", q or ""):
        t = (m.group(0) or "").strip().lower()
        if t and t not in toks:
            toks.append(t)
    # Also include words >= 4 chars from raw text for message-like queries
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_]{3,}", q or ""):
        t = (m.group(0) or "").strip().lower()
        if t and t not in toks:
            toks.append(t)
    return toks[:limit]


def _literal_text(n: ast.AST) -> str:
    # Extract visible text from Constant(str) and JoinedStr (f-strings)
    if isinstance(n, ast.Constant) and isinstance(n.value, str):
        return n.value
    if isinstance(n, ast.JoinedStr):
        parts: List[str] = []
        for v in n.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            # skip FormattedValue dynamic expressions
        return "".join(parts)
    return ""


def stage_pyliterals_hits(query: str, k: int, *, max_time_ms: int | None = 200) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: scan Python string literals (including f-strings) for message-like queries.

    Useful for locating error messages, log messages, and textual fragments that users often paste.
    """
    q = (query or "").strip()
    if not q:
        return []
    toks = _query_tokens(q)
    qn = _norm(q)
    if not toks and len(qn) < 3:
        return []

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

        best: Tuple[float, int, int] | None = None
        class V(ast.NodeVisitor):
            def visit_Constant(self, node: ast.Constant) -> None:  # type: ignore[name-defined]
                self._consider(node)
            def visit_JoinedStr(self, node: ast.JoinedStr) -> None:  # type: ignore[name-defined]
                self._consider(node)
            def _consider(self, node: ast.AST) -> None:
                if time_up():
                    return
                s = _literal_text(node)
                if not s:
                    return
                sn = _norm(s)
                score = 0.0
                # token containment
                if toks and any(t in sn for t in toks):
                    score += 1.0
                # fuzzy similarity (light)
                try:
                    import difflib as _df
                    sim = _df.SequenceMatcher(None, qn, sn).ratio()
                    if sim >= 0.80:
                        score += 0.7
                except Exception:
                    pass
                if score > 0.0:
                    ln = getattr(node, "lineno", 0) or 0
                    if ln > 0:
                        a = max(1, ln - 12)
                        b = min(len(lines), ln + 12)
                        nonlocal best
                        cand = (score, a, b)
                        if best is None or cand[0] > best[0]:
                            best = cand
        V().visit(tree)
        if best is not None:
            sc, a, b = best
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
            # Priority near docstrings, below pyflow/libcst
            hits.append((0.9915, rel_p, obj))
            if len(hits) >= k:
                break

    return hits


__all__ = ["stage_pyliterals_hits"]
