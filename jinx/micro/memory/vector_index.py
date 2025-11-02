from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

from jinx.micro.embeddings.retrieval import retrieve_top_k as _emb_retrieve
from jinx.micro.embeddings.text_clean import is_noise_text as _is_noise


async def search(
    query: str,
    *,
    k: int = 8,
    scope: str = "any",
    max_time_ms: int = 45,
    preview_chars: int = 160,
) -> List[str]:
    """Vector-based memory search (async, RT-bounded).

    Reuses the existing embeddings runtime store. Prefers memory-ingested lines
    (meta.kind == "mem" or meta.source == "state"), then fills from other
    allowed sources if needed. Returns unique, trimmed previews.

    Parameters
    ----------
    query : str
        Natural language query.
    k : int
        Number of lines to return (cap). Default 8.
    scope : str
        Currently unused hint (reserved for future per-level indexes).
    max_time_ms : int
        Time budget for retrieval. Default 45ms.
    preview_chars : int
        Max characters per returned line. Default 160.
    """
    q = (query or "").strip()
    if not q:
        return []

    # Over-fetch slightly to have room for filtering/dedup.
    try:
        overfetch = max(k * 3, int(os.getenv("JINX_MEM_VEC_OVERFETCH", str(k * 4))))
    except Exception:
        overfetch = k * 4

    try:
        hits: List[Tuple[float, str, Dict[str, Any]]] = await _emb_retrieve(q, k=overfetch, max_time_ms=max_time_ms)
    except Exception:
        hits = []
    if not hits:
        return []

    def _is_mem(meta: Dict[str, Any]) -> bool:
        src = (meta.get("source") or "").strip().lower()
        kind = (meta.get("kind") or "").strip().lower()
        return kind == "mem" or src == "state"

    # Prefer memory-ingested items first, preserve per-bucket order.
    mem_hits: List[Tuple[float, str, Dict[str, Any]]] = []
    other_hits: List[Tuple[float, str, Dict[str, Any]]] = []
    for sc, src, obj in hits:
        meta = obj.get("meta", {})
        if _is_mem(meta):
            mem_hits.append((sc, src, obj))
        else:
            other_hits.append((sc, src, obj))

    ordered = mem_hits + other_hits

    out: List[str] = []
    seen: set[str] = set()
    for _sc, _src, obj in ordered:
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        if not pv or pv in seen or _is_noise(pv):
            continue
        out.append(pv[:preview_chars])
        seen.add(pv)
        if len(out) >= k:
            break
    return out[:k]


async def upsert(lines: List[str]) -> None:
    """Optional ingestion helper.

    Currently a no-op placeholder since `indexer.ingest_memory` handles ingestion.
    Left for future backends (e.g., dedicated FAISS or remote stores).
    """
    return None
