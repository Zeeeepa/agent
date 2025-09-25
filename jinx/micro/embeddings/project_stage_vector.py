from __future__ import annotations

import time
import asyncio
from typing import Any, Dict, List, Tuple

from .project_query_embed import embed_query
from .project_scan_store import iter_project_chunks
from .project_retrieval_config import (
    PROJ_MIN_PREVIEW_LEN,
    PROJ_SCORE_THRESHOLD,
    PROJ_MAX_FILES,
    PROJ_MAX_CHUNKS_PER_FILE,
)
from .util import cos


async def stage_vector_hits(query: str, k: int, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage 1: vector similarity search over project chunks.

    Returns a list of (score, file_rel, obj) sorted by score desc.
    """
    q = (query or "").strip()
    if not q:
        return []
    qv = await embed_query(q)

    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    t0 = time.perf_counter()
    n = 0
    batch_size = 64

    # Create a generator for chunks
    it = iter_project_chunks(max_files=PROJ_MAX_FILES, max_chunks_per_file=PROJ_MAX_CHUNKS_PER_FILE)

    def _next_batch(gen, nmax: int):
        out = []
        for _ in range(nmax):
            try:
                out.append(next(gen))
            except StopIteration:
                break
        return out

    while True:
        # Fetch a batch of chunks off the event loop
        batch = await asyncio.to_thread(_next_batch, it, batch_size)
        if not batch:
            break
        for file_rel, obj in batch:
            vec = obj.get("embedding") or []
            meta = obj.get("meta", {})
            pv = (meta.get("text_preview") or '').strip()
            if len(pv) < PROJ_MIN_PREVIEW_LEN:
                continue
            s = cos(qv, vec)
            if s < PROJ_SCORE_THRESHOLD:
                continue
            scored.append((s, str(meta.get("file_rel") or file_rel or ''), obj))
            n += 1
        # Yield between batches to keep UI responsive
        await asyncio.sleep(0)
        if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            break
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
