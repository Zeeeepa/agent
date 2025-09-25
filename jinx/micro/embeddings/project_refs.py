from __future__ import annotations

import os
from typing import List, Tuple

from .project_scan_store import iter_project_chunks
from .project_config import ROOT, INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_line_window import find_line_window
from .project_lang import lang_for_file
from .project_iter import iter_candidate_files


def find_usages_in_project(symbol: str, exclude_rel: str, *, limit: int = 3, around: int = 8, scan_cap_files: int = 800) -> List[Tuple[str, int, int, str, str]]:
    """Find up to `limit` usages of `symbol` across the project.

    Returns a list of (file_rel, line_start, line_end, snippet, lang).
    Best-effort: scans up to `scan_cap_files` files in total; dedupes by file.
    """
    sym = (symbol or "").strip()
    if not sym:
        return []
    got: List[Tuple[str, int, int, str, str]] = []
    seen_files: set[str] = set()
    n = 0
    # Pass 1: prefer files already known to embeddings store (fast)
    for file_rel, _obj in iter_project_chunks():
        n += 1
        if n > scan_cap_files:
            break
        fr = (file_rel or "").strip()
        if not fr or fr == exclude_rel or fr in seen_files:
            continue
        abs_path = os.path.join(ROOT, fr)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        a, b, snip = find_line_window(text, [sym], around=around)
        if a or b:
            seen_files.add(fr)
            lang = lang_for_file(fr)
            got.append((fr, a, b, snip, lang))
            if len(got) >= limit:
                return got
    # Pass 2: fallback to scanning candidate project files not yet covered
    if len(got) < limit:
        for abs_p, rel_p in iter_candidate_files(
            ROOT,
            include_exts=INCLUDE_EXTS,
            exclude_dirs=EXCLUDE_DIRS,
            max_file_bytes=MAX_FILE_BYTES,
        ):
            if n > scan_cap_files:
                break
            n += 1
            fr = (rel_p or "").strip()
            if not fr or fr == exclude_rel or fr in seen_files:
                continue
            try:
                with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue
            a, b, snip = find_line_window(text, [sym], around=around)
            if a or b:
                seen_files.add(fr)
                lang = lang_for_file(fr)
                got.append((fr, a, b, snip, lang))
                if len(got) >= limit:
                    break
    return got
