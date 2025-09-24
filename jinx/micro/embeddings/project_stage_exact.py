from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

from .project_scan_store import iter_project_chunks
from .project_retrieval_config import (
    PROJ_MAX_FILES,
    PROJ_MAX_CHUNKS_PER_FILE,
)
from .project_config import ROOT
from .project_query_tokens import expand_strong_tokens


def stage_exact_hits(query: str, k: int, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage 0: exact/identifier substring matches in preview/terms/path.

    Returns a list of (score, file_rel, obj) sorted by score desc.
    """
    q = (query or "").strip()
    if not q:
        return []
    code_toks = expand_strong_tokens(q, max_items=32)
    code_toks = code_toks[:8]
    if not code_toks:
        return []
    t0 = time.perf_counter()
    exact_hits: List[Tuple[float, str, Dict[str, Any]]] = []
    seen_files: set[str] = set()
    for file_rel, obj in iter_project_chunks(max_files=PROJ_MAX_FILES, max_chunks_per_file=PROJ_MAX_CHUNKS_PER_FILE):
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "")
        hay_terms = " ".join([str(x) for x in (meta.get("terms") or [])])
        hay = (pv + " " + hay_terms + " " + str(file_rel)).lower()
        ok = any(t.lower() in hay for t in code_toks)
        # If not matched in preview/terms/path, try a single read of the file text for this file
        if not ok and file_rel and file_rel not in seen_files:
            seen_files.add(file_rel)
            try:
                with open(os.path.join(ROOT, file_rel), 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
            except Exception:
                txt = ""
            if txt:
                low = txt.lower()
                if any(t.lower() in low for t in code_toks):
                    ok = True
        if not ok:
            continue
        exact_hits.append((0.95, str(meta.get("file_rel") or file_rel or ""), obj))
        if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            break
    if not exact_hits:
        return []
    exact_hits.sort(key=lambda x: x[0], reverse=True)
    return exact_hits[:k]
