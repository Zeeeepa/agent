from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import re as _re

from .project_scan_store import iter_project_chunks
from .project_retrieval_config import (
    PROJ_MAX_FILES,
    PROJ_MAX_CHUNKS_PER_FILE,
)


def stage_keyword_hits(query: str, k: int, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage 2: lightweight keyword/substring fallback over preview, terms, and path.

    Returns a list of (score, file_rel, obj) sorted by score desc.
    """
    q = (query or "").strip()
    if not q:
        return []
    toks = [t.lower() for t in _re.findall(r"(?u)[\w\.]+", q) if len(t) >= 2]
    if not toks:
        return []
    t0 = time.perf_counter()
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for file_rel, obj in iter_project_chunks(max_files=PROJ_MAX_FILES, max_chunks_per_file=PROJ_MAX_CHUNKS_PER_FILE):
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").lower()
        terms = [str(x).lower() for x in (meta.get("terms") or [])]
        hay = " ".join([pv, str(file_rel).lower(), " ".join(terms)])
        hits = sum(1 for t in toks if t in hay)
        if hits <= 0:
            continue
        s = 0.18 + min(0.04 * hits, 0.20)
        scored.append((s, str(meta.get("file_rel") or file_rel or ""), obj))
        if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            break
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
