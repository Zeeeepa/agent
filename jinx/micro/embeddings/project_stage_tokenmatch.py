from __future__ import annotations

import io
import os
import time
import tokenize
from typing import Any, Dict, List, Tuple
import re as _re

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks


def _time_up(t0: float, limit_ms: int | None) -> bool:
    return limit_ms is not None and (time.perf_counter() - t0) * 1000.0 > limit_ms


_NAME_RE = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_STOP = {
    "and", "or", "not", "is", "in", "if", "else", "elif", "for", "while", "return",
    "class", "def", "with", "as", "try", "except", "finally", "lambda", "True", "False", "None",
}


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


def _significant_tokens(tok_pairs: List[Tuple[str, Tuple[int, int]]]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for s, (ln, _col) in tok_pairs:
        if not s:
            continue
        if not _NAME_RE.match(s):
            continue
        if s in _STOP:
            continue
        out.append((s, int(ln or 0)))
    return out


def _match_ordered_subsequence(hay: List[Tuple[str, int]], needle: List[str]) -> Tuple[int, int] | None:
    """Return (start_line, end_line) covering an ordered subsequence match, allowing gaps.

    Chooses the first span that matches all needle tokens in order. Lines are derived
    from the first and last matched tokens.
    """
    if not hay or not needle:
        return None
    hi = 0
    start_line = end_line = 0
    # find first
    while hi < len(hay) and hay[hi][0] != needle[0]:
        hi += 1
    if hi >= len(hay):
        return None
    start_line = hay[hi][1]
    # advance for the rest
    ni = 1
    hi += 1
    while ni < len(needle) and hi < len(hay):
        if hay[hi][0] == needle[ni]:
            ni += 1
            end_line = hay[hi][1]
        hi += 1
    if ni < len(needle):
        return None
    if end_line <= 0:
        end_line = start_line
    return (start_line, end_line)


def stage_tokenmatch_hits(query: str, k: int, *, max_time_ms: int | None = 200) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Match query's Python token sequence as an exact subsequence in project .py files.

    This is whitespace-agnostic and comments-agnostic, robust to formatting changes.
    """
    q = (query or "").strip()
    if not q:
        return []
    t0 = time.perf_counter()

    # Tokenize query and filter to significant tokens
    q_toks = _tokenize_code(q)
    q_sig = _significant_tokens(q_toks)
    q_vals = [s for s, _ln in q_sig]
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
        hay_sig = _significant_tokens(toks)
        span = _match_ordered_subsequence(hay_sig, q_vals)
        if span is None:
            return False
        s_line, e_line = span
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
