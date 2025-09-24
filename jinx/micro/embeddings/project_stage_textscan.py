from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_line_window import find_line_window
from .project_scan_store import iter_project_chunks
from .project_query_tokens import expand_strong_tokens, codeish_tokens


def _expand_tokens(q: str, max_items: int = 32) -> List[str]:
    strong = expand_strong_tokens(q, max_items=max_items)
    simple = codeish_tokens(q)
    # Deduplicate preserving order, prefer strong first
    out: List[str] = []
    seen: set[str] = set()
    for t in strong + simple:
        tl = (t or "").lower()
        if not tl or tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out[:max_items]


def stage_textscan_hits(query: str, k: int, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage -1: direct text scan over project files (no embeddings).

    Returns a list of (score, file_rel, obj) sorted by score desc.
    """
    q = (query or "").strip()
    if not q:
        return []
    toks = _expand_tokens(q)
    if not toks:
        return []

    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []
    seen_rel: set[str] = set()

    # Pass 1: only files already present in embeddings store (fast and highly relevant)
    rel_files: List[str] = []
    try:
        seen_f: set[str] = set()
        for fr, obj in iter_project_chunks():
            rel = fr or str((obj.get("meta") or {}).get("file_rel") or "")
            if rel and rel not in seen_f:
                seen_f.add(rel)
                rel_files.append(rel)
    except Exception:
        rel_files = []

    def _scan_abs_rel(abs_p: str, rel_p: str) -> bool:
        # Basic time budget check
        if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            return True  # signal to stop
        # Avoid scanning the same rel path twice (shouldn't happen, but be safe)
        if rel_p in seen_rel:
            return False
        seen_rel.add(rel_p)
        try:
            with open(abs_p, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            text = ""
        if not text:
            return False
        low = text.lower()
        if not any(t.lower() in low for t in toks):
            return False
        # Small window as preview; snippet builder will expand to full scope later for Python
        a, b, snip = find_line_window(text, toks, around=12)
        # If multiple occurrences across the file, escalate meta to whole-file range
        multi_in_text = any(text.lower().count(t.lower()) >= 2 for t in toks)
        ls_meta = int(a or 0)
        le_meta = int(b or 0)
        if multi_in_text:
            ls_meta = 1
            le_meta = len(text.splitlines())
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": snip or text[:300].strip(),
                "line_start": ls_meta,
                "line_end": le_meta,
            },
        }
        hits.append((0.98, rel_p, obj))
        if len(hits) >= k:
            return True
        return False
    # Scan embeddings-known files first
    for rel_p in rel_files:
        abs_p = os.path.join(ROOT, rel_p)
        if _scan_abs_rel(abs_p, rel_p):
            return hits[:k]

    # Pass 2: fallback to general project walk (slower)
    for abs_p, rel_p in iter_candidate_files(
        ROOT,
        include_exts=INCLUDE_EXTS,
        exclude_dirs=EXCLUDE_DIRS,
        max_file_bytes=MAX_FILE_BYTES,
    ):
        if _scan_abs_rel(abs_p, rel_p):
            return hits[:k]

    # Keep original order; all scores equal
    return hits[:k]
