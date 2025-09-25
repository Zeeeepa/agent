from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks
from .project_query_tokens import expand_strong_tokens, codeish_tokens

try:
    # RapidFuzz is preferred over fuzzywuzzy (faster, no Levenshtein dep requirement)
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dep
    fuzz = None  # type: ignore


def _rf_partial_ratio(a: str, b: str) -> float:
    """Safe RapidFuzz partial_ratio wrapper returning [0.0,1.0].

    Avoids direct attribute access on a possibly-None module for static analyzers.
    """
    try:
        # Import inside to ensure availability and keep type-checkers happy
        from rapidfuzz import fuzz as _fuzz  # type: ignore
        return float(_fuzz.partial_ratio(a, b)) / 100.0
    except Exception:
        return 0.0


def _anchors(q: str, limit: int = 6) -> List[str]:
    toks: List[str] = []
    toks.extend(expand_strong_tokens(q, max_items=32))
    for t in codeish_tokens(q):
        if t not in toks:
            toks.append(t)
    # Drop trivial tokens
    bad = {"for", "in", "def", "class", "return", "async", "await"}
    toks2 = [t for t in toks if len(t) >= 3 and t.lower() not in bad]
    # Prefer longer unique tokens
    seen: set[str] = set()
    out: List[str] = []
    for t in sorted(toks2, key=len, reverse=True):
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out[:limit]


def stage_rapidfuzz_hits(query: str, k: int, *, max_time_ms: int | None = 240) -> List[Tuple[float, str, Dict[str, Any]]]:
    """RapidFuzz-based approximate matching for code fragments.

    Strategy:
    - Build a small set of anchors from the query (used only as a prefilter to keep costs low).
    - Score overlapping multi-line windows near lines containing anchors using fuzz.partial_ratio.
    - Return top-k windows within time budget.
    """
    if fuzz is None:
        return []
    q = (query or "").strip()
    if not q:
        return []
    t0 = time.perf_counter()

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    anchors = [a.lower() for a in _anchors(q, limit=5)]
    # Window config approximates query size
    qlen = max(20, min(800, len(q)))
    win_chars = int(min(max(60, qlen * 2), 1500))
    step_lines = 6  # coarse stride for speed

    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def consider_window(rel_p: str, lines: List[str], a: int, b: int) -> None:
        if a < 1 or b > len(lines) or a > b:
            return
        snippet = "\n".join(lines[a - 1 : b]).strip()
        if not snippet:
            return
        # Score snippet against query using partial_ratio (fast and robust)
        score = _rf_partial_ratio(q, snippet)
        if score < 0.90:  # stricter threshold for precision
            return
        # Clamp so that exact/token/AST matchers always outrank RapidFuzz
        if score > 0.986:
            score = 0.986
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": snippet,
                "line_start": a,
                "line_end": b,
            },
        }
        hits.append((score, rel_p, obj))

    def scan_text(rel_p: str, text: str) -> None:
        lines = text.splitlines()
        lo_lines = [ln.lower() for ln in lines]
        n = len(lines)
        if n == 0:
            return
        # Anchor prefilter: collect candidate line indices containing at least one anchor
        cand_idxs: List[int] = []
        if anchors:
            for i, ln in enumerate(lo_lines, start=1):
                for a in anchors:
                    if a in ln:
                        cand_idxs.append(i)
                        break
        else:
            # If no anchors, sample lines with a stride to bound runtime
            cand_idxs = list(range(1, n + 1, max(1, step_lines)))
        # Expand around candidate indices with rough char-sized windows
        for i in cand_idxs:
            if time_up():
                return
            # Choose window lines around i to approximate win_chars
            # Expand outwards until reaching char budget or file edges
            a = i
            b = i
            cur = len(lines[i - 1])
            while (a > 1 or b < n) and cur < win_chars:
                left_len = len(lines[a - 2]) if a > 1 else 10**9
                right_len = len(lines[b]) if b < n else 10**9
                if left_len <= right_len and a > 1:
                    a -= 1
                    cur += left_len + 1
                elif b < n:
                    b += 1
                    cur += right_len + 1
                else:
                    break
            consider_window(rel_p, lines, a, b)

    def process(abs_p: str, rel_p: str) -> bool:
        if time_up():
            return True
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return False
        if not text:
            return False
        scan_text(rel_p, text)
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
                return sorted(hits, key=lambda x: -x[0])[:k]
    except Exception:
        pass

    # Pass 2: general walk over .py files
    for ap, rel in iter_candidate_files(ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if process(ap, rel):
            return sorted(hits, key=lambda x: -x[0])[:k]

    return sorted(hits, key=lambda x: -x[0])[:k]


__all__ = ["stage_rapidfuzz_hits"]
