from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks
from .project_query_tokens import expand_strong_tokens, codeish_tokens


def _anchors(q: str, limit: int = 4) -> List[str]:
    toks: List[str] = []
    toks.extend(expand_strong_tokens(q, max_items=32))
    for t in codeish_tokens(q):
        if t not in toks:
            toks.append(t)
    # drop trivial words
    bad = {"for", "in", "def", "class", "return", "async", "await"}
    # prefer longer tokens
    toks2 = [t for t in toks if len(t) >= 3 and t.lower() not in bad]
    seen: set[str] = set()
    out: List[str] = []
    for t in sorted(toks2, key=len, reverse=True):
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out[:limit]


def stage_linetokens_hits(query: str, k: int, *, max_time_ms: int | None = 120) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: scan for lines that contain all anchor tokens (case-insensitive).

    Extremely cheap line-level matcher; good for short assignment/comprehension queries.
    """
    q = (query or "").strip()
    if not q:
        return []
    t0 = time.perf_counter()

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    anchors = [a.lower() for a in _anchors(q, limit=3)]
    if not anchors:
        return []

    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def process(abs_p: str, rel_p: str) -> bool:
        if time_up():
            return True
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
        except Exception:
            return False
        if not lines:
            return False
        lo_lines = [ln.lower() for ln in lines]
        for idx, ln in enumerate(lo_lines):
            if all(a in ln for a in anchors):
                a = max(1, idx + 1 - 12)
                b = min(len(lines), idx + 1 + 12)
                snip = "\n".join(lines[a-1:b]).strip()
                obj = {
                    "embedding": [],
                    "meta": {
                        "file_rel": rel_p,
                        "text_preview": snip or "\n".join(lines[max(0, idx-1):min(len(lines), idx+2)]).strip(),
                        "line_start": a,
                        "line_end": b,
                    },
                }
                hits.append((0.995, rel_p, obj))
                if len(hits) >= k:
                    return True
        return False

    # embeddings-known files first
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

    # full walk
    for ap, rel in iter_candidate_files(ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if process(ap, rel):
            return hits[:k]

    return hits[:k]


__all__ = ["stage_linetokens_hits"]
