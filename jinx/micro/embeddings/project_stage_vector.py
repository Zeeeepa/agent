from __future__ import annotations

import time
import os
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
from .hot_store import get_project_chunks_hot
from .similarity import score_cosine_batch
from .ann_index import search_ann_items


async def stage_vector_hits(query: str, k: int, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage 1: vector similarity search over project chunks.

    Returns a list of (score, file_rel, obj) sorted by score desc.
    """
    q = (query or "").strip()
    if not q:
        return []
    # Overlap query embedding with loading a hot snapshot of chunks (TTL-based)
    try:
        ttl_ms = int(os.getenv("EMBED_PROJECT_HOT_TTL_MS", "1600"))
    except Exception:
        ttl_ms = 1600

    async def _load_all() -> List[Tuple[str, Dict[str, Any]]]:
        def _work() -> List[Tuple[str, Dict[str, Any]]]:
            out: List[Tuple[str, Dict[str, Any]]] = []
            for pair in iter_project_chunks(max_files=PROJ_MAX_FILES, max_chunks_per_file=PROJ_MAX_CHUNKS_PER_FILE):
                out.append(pair)
            return out
        return await asyncio.to_thread(_work)

    qv_task = asyncio.create_task(embed_query(q))
    items_task = asyncio.create_task(get_project_chunks_hot(_load_all, ttl_ms))
    qv = await qv_task
    items = await items_task
    # Prefer ANN candidate generation (with brute-force fallback inside)
    try:
        try:
            overfetch = int(os.getenv("EMBED_PROJECT_ANN_OVERFETCH", "5"))
        except Exception:
            overfetch = 5
        top_n = max(k, min(len(items), max(100, k * max(1, overfetch))))
        t0 = time.perf_counter()

        def _rank_candidates() -> List[Tuple[int, float]]:
            return search_ann_items(qv, items, top_n, max_time_ms=max_time_ms)

        pairs = await asyncio.to_thread(_rank_candidates)
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for idx, s in pairs:
            try:
                file_rel_i, obj_i = items[idx]
            except Exception:
                continue
            if float(s or 0.0) < PROJ_SCORE_THRESHOLD:
                continue
            meta_i = obj_i.get("meta", {})
            pv = (meta_i.get("text_preview") or '').strip()
            if len(pv) < PROJ_MIN_PREVIEW_LEN:
                continue
            scored.append((float(s or 0.0), str(meta_i.get("file_rel") or file_rel_i or ''), obj_i))
            if len(scored) >= k and (max_time_ms is not None):
                # Respect overall budget roughly
                if (time.perf_counter() - t0) * 1000.0 > max_time_ms:
                    break
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
    except Exception:
        # Defensive fallback to previous brute-force batch path
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        t0 = time.perf_counter()
        batch_vecs: List[List[float]] = []
        batch_meta: List[Tuple[str, Dict[str, Any]]] = []  # (file_rel, obj)
        BATCH = 512

        async def _flush_batch() -> None:
            nonlocal batch_vecs, batch_meta, scored
            if not batch_vecs:
                return
            sims = score_cosine_batch(qv, batch_vecs)
            for (file_rel_i, obj_i), s in zip(batch_meta, sims):
                if s < PROJ_SCORE_THRESHOLD:
                    continue
                meta_i = obj_i.get("meta", {})
                pv = (meta_i.get("text_preview") or '').strip()
                if len(pv) < PROJ_MIN_PREVIEW_LEN:
                    continue
                scored.append((s, str(meta_i.get("file_rel") or file_rel_i or ''), obj_i))
            batch_vecs = []
            batch_meta = []

        for idx, (file_rel, obj) in enumerate(items):
            vec = obj.get("embedding") or []
            meta = obj.get("meta", {})
            pv = (meta.get("text_preview") or '').strip()
            if len(pv) < PROJ_MIN_PREVIEW_LEN:
                continue
            batch_vecs.append(vec)
            batch_meta.append((file_rel, obj))
            if len(batch_vecs) >= BATCH:
                await _flush_batch()
                await asyncio.sleep(0)
            if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
                break
        await _flush_batch()
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
