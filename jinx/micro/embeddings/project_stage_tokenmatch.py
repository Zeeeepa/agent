from __future__ import annotations

import io
import os
import time
import tokenize
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks


def _time_up(t0: float, limit_ms: int | None) -> bool:
    return limit_ms is not None and (time.perf_counter() - t0) * 1000.0 > limit_ms


def _tokenize_code(src: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Return a list of (token_string, (line, col)) for significant tokens in Python code.

    Ignores whitespace, comments, encoding tokens, and indentation artifacts.
    """
    try:
        data = src.encode("utf-8")
    except Exception:
        return []
    toks: List[Tuple[str, Tuple[int, int]]] = []
    try:
        for tok in tokenize.tokenize(io.BytesIO(data).readline):
            ttype = tok.type
            s = tok.string
            if ttype in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.COMMENT):
                continue
            if not s:
                continue
            toks.append((s, tok.start))
    except Exception:
        return []
    return toks


def _match_subsequence(hay: List[str], needle: List[str]) -> int | None:
    """Return starting index of needle in hay as exact subsequence, or None."""
    if not hay or not needle or len(needle) > len(hay):
        return None
    first = needle[0]
    max_i = len(hay) - len(needle)
    i = 0
    while i <= max_i:
        # fast skip to next candidate
        try:
            j = hay.index(first, i)
        except ValueError:
            return None
        # verify
        ok = True
        for k in range(1, len(needle)):
            if hay[j + k] != needle[k]:
                ok = False
                break
        if ok:
            return j
        i = j + 1
    return None


def stage_tokenmatch_hits(query: str, k: int, *, max_time_ms: int | None = 200) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Match query's Python token sequence as an exact subsequence in project .py files.

    This is whitespace-agnostic and comments-agnostic, robust to formatting changes.
    """
    q = (query or "").strip()
    if not q:
        return []
    t0 = time.perf_counter()

    # Tokenize query
    q_toks = _tokenize_code(q)
    q_vals = [s for s, _pos in q_toks]
    if not q_vals:
        return []

    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def process(abs_p: str, rel_p: str) -> bool:
        if _time_up(t0, max_time_ms):
            return True
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except Exception:
            return False
        if not txt:
            return False
        toks = _tokenize_code(txt)
        if not toks:
            return False
        hay = [s for s, _pos in toks]
        idx = _match_subsequence(hay, q_vals)
        if idx is None:
            return False
        # Map token index range to line window
        s_line = toks[idx][1][0]
        e_line = toks[min(len(toks) - 1, idx + len(q_vals) - 1)][1][0]
        lines = txt.splitlines()
        a = max(1, s_line - 12)
        b = min(len(lines), e_line + 12)
        snip = "\n".join(lines[a-1:b]).strip()
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": snip or "\n".join(lines[max(0, s_line-1):min(len(lines), e_line)]).strip(),
                "line_start": a,
                "line_end": b,
            },
        }
        hits.append((0.999, rel_p, obj))
        return len(hits) >= k

    # Pass 1: embeddings-known files first
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

    # Pass 2: general walk
    for ap, rel in iter_candidate_files(ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if process(ap, rel):
            return hits[:k]

    return hits[:k]


__all__ = ["stage_tokenmatch_hits"]
