from __future__ import annotations

import asyncio
import time
import os
from typing import List, Tuple, Dict, Any
import hashlib

from jinx.micro.embeddings.pipeline import iter_recent_items
from .paths import EMBED_ROOT
from .similarity import score_cosine_batch
from .text_clean import is_noise_text
from .scan_store import iter_items as scan_iter_items
from .embed_cache import embed_text_cached
from .ann_index_runtime import search_ann_items as _search_ann_runtime

DEFAULT_TOP_K = int(os.getenv("EMBED_TOP_K", "5"))
# Balanced defaults; adapt at runtime based on query length
SCORE_THRESHOLD = float(os.getenv("EMBED_SCORE_THRESHOLD", "0.25"))
MIN_PREVIEW_LEN = int(os.getenv("EMBED_MIN_PREVIEW_LEN", "8"))
MAX_FILES_PER_SOURCE = int(os.getenv("EMBED_MAX_FILES_PER_SOURCE", "500"))
MAX_SOURCES = int(os.getenv("EMBED_MAX_SOURCES", "50"))
QUERY_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RECENCY_WINDOW_SEC = int(os.getenv("EMBED_RECENCY_WINDOW_SEC", str(24 * 3600)))
EXHAUSTIVE = str(os.getenv("EMBED_EXHAUSTIVE", "1")).lower() in {"1", "true", "on", "yes"}

# Shared hot-store for runtime items
try:
    _HOT_TTL_MS = int(os.getenv("EMBED_RUNTIME_HOT_TTL_MS", "1500"))
except Exception:
    _HOT_TTL_MS = 1500
from .hot_store import get_runtime_items_hot

async def _load_runtime_items() -> List[Tuple[str, Dict[str, Any]]]:
    return await asyncio.to_thread(scan_iter_items, EMBED_ROOT, MAX_FILES_PER_SOURCE, MAX_SOURCES)


async def _embed_query(text: str) -> List[float]:
    try:
        # Shared cached embedding call with TTL, coalescing, concurrency limit and timeout
        return await embed_text_cached(text, model=QUERY_MODEL)
    except Exception:
        # Best-effort: return empty vector on API failure
        return []


def _iter_items() -> List[Tuple[str, Dict[str, Any]]]:
    # Delegate on-disk scanning to a dedicated helper for clarity and reuse
    return scan_iter_items(EMBED_ROOT, MAX_FILES_PER_SOURCE, MAX_SOURCES)


async def retrieve_top_k(query: str, k: int | None = None, *, max_time_ms: int | None = 200) -> List[Tuple[float, str, Dict[str, Any]]]:
    # Adapt parameters based on query length (short queries get lower threshold and higher k)
    q = (query or "").strip()
    qlen = len(q)
    thr = SCORE_THRESHOLD
    k_eff = k or DEFAULT_TOP_K
    if qlen <= 12:
        thr = max(0.15, thr * 0.8)
        k_eff = max(k_eff, 8)
    elif qlen <= 24:
        thr = max(0.2, thr)
        k_eff = max(k_eff, 6)

    # Overlap query embedding with a hot-store refresh to reduce wall time
    qv_task = asyncio.create_task(_embed_query(query))
    hot_task = asyncio.create_task(get_runtime_items_hot(_load_runtime_items, _HOT_TTL_MS))
    qv = await qv_task
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    now = time.time()
    t0 = time.perf_counter()

    # 1) Fast-path: score in-memory recent items first
    try:
        state_boost = max(1.0, float(os.getenv("EMBED_STATE_BOOST", "1.1")))
    except Exception:
        state_boost = 1.1
    try:
        state_rec_mult = max(0.0, float(os.getenv("EMBED_STATE_RECENCY_MULT", "0.5")))
    except Exception:
        state_rec_mult = 0.5
    short_q = (qlen <= int(os.getenv("JINX_CONTINUITY_SHORTLEN", "80")))
    _recent_objs: List[Dict[str, Any]] = []
    _recent_vecs: List[List[float]] = []
    _recent_meta: List[Dict[str, Any]] = []
    for obj in iter_recent_items():
        meta = obj.get("meta", {})
        src_l = (meta.get("source") or "").strip().lower()
        if not (src_l == "dialogue" or src_l.startswith("sandbox/") or src_l == "state"):
            continue
        pv = (meta.get("text_preview") or "").strip()
        if len(pv) < MIN_PREVIEW_LEN or is_noise_text(pv):
            continue
        _recent_objs.append(obj)
        _recent_meta.append(meta)
        _recent_vecs.append(obj.get("embedding") or [])
        if len(_recent_objs) >= k_eff * 2:  # cap to a small multiple of k
            break
    if _recent_vecs:
        sims = score_cosine_batch(qv, _recent_vecs)
        for obj, meta, sim in zip(_recent_objs, _recent_meta, sims):
            if sim < thr:
                continue
            ts = float(meta.get("ts") or 0.0)
            age = max(0.0, now - ts)
            rec = 0.0 if RECENCY_WINDOW_SEC <= 0 else max(0.0, 1.0 - (age / RECENCY_WINDOW_SEC))
            score = 0.8 * sim + 0.2 * rec
            if (meta.get("source") or "").strip().lower() == "state" and short_q:
                score *= state_boost * (1.0 + state_rec_mult * rec)
            scored.append((score, meta.get("source", "recent"), obj))
            if len(scored) >= k_eff:
                break

    if len(scored) >= k_eff:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k_eff]
    eff_budget = None if EXHAUSTIVE else max_time_ms

    # 2) Persisted items: filter by source and preview, then ANN overlay (fallback to batch cosine)
    items_all = await hot_task
    items: List[Tuple[str, Dict[str, Any]]] = []
    for src, obj in (items_all or []):
        meta = (obj or {}).get("meta", {})
        src_l = (src or "").strip().lower()
        meta_src_l = (meta.get("source") or "").strip().lower()
        allow_src = (
            src_l == "dialogue" or src_l.startswith("sandbox/") or src_l == "state" or
            meta_src_l == "dialogue" or meta_src_l.startswith("sandbox/") or meta_src_l == "state"
        )
        if not allow_src:
            continue
        pv = (meta.get("text_preview") or "").strip()
        if len(pv) < MIN_PREVIEW_LEN or is_noise_text(pv):
            continue
        items.append((src, obj))

    # ANN candidate generation
    try:
        overfetch = max(k_eff * 4, int(os.getenv("EMBED_RUNTIME_ANN_OVERFETCH", str(k_eff * 6))))
    except Exception:
        overfetch = k_eff * 6
    try:
        def _rank_candidates() -> List[Tuple[int, float]]:
            return _search_ann_runtime(qv, items, top_n=min(len(items), max(k_eff, overfetch)))
        scored_candidates = await asyncio.to_thread(_rank_candidates)
    except Exception:
        scored_candidates = []  # type: ignore[name-defined]

    if scored_candidates:
        for idx_c, sim in scored_candidates:
            if idx_c < 0 or idx_c >= len(items):
                continue
            src_i, obj_i = items[idx_c]
            if float(sim or 0.0) < thr:
                continue
            ts = float(obj_i.get("meta", {}).get("ts") or 0.0)
            age = max(0.0, now - ts)
            rec = 0.0 if RECENCY_WINDOW_SEC <= 0 else max(0.0, 1.0 - (age / RECENCY_WINDOW_SEC))
            score = 0.8 * sim + 0.2 * rec
            if (obj_i.get("meta", {}).get("source") or "").strip().lower() == "state" and short_q:
                score *= state_boost * (1.0 + state_rec_mult * rec)
            scored.append((score, obj_i.get("meta", {}).get("source", "persisted"), obj_i))
            if len(scored) >= k_eff:
                break
            if eff_budget is not None and (time.perf_counter() - t0) * 1000.0 > eff_budget:
                break
    else:
        # Batch cosine fallback
        B = 1024
        buf_vecs: List[List[float]] = []
        buf_meta_src: List[Tuple[str, Dict[str, Any]]] = []
        def _flush() -> None:
            nonlocal scored, buf_vecs, buf_meta_src
            if not buf_vecs:
                return
            sims = score_cosine_batch(qv, buf_vecs)
            for (src_i, obj_i), sim in zip(buf_meta_src, sims):
                if sim < thr:
                    continue
                ts = float(obj_i.get("meta", {}).get("ts") or 0.0)
                age = max(0.0, now - ts)
                rec = 0.0 if RECENCY_WINDOW_SEC <= 0 else max(0.0, 1.0 - (age / RECENCY_WINDOW_SEC))
                score = 0.8 * sim + 0.2 * rec
                if (obj_i.get("meta", {}).get("source") or "").strip().lower() == "state" and short_q:
                    score *= state_boost * (1.0 + state_rec_mult * rec)
                scored.append((score, obj_i.get("meta", {}).get("source", "persisted"), obj_i))
            buf_vecs = []
            buf_meta_src = []
        for idx, (src, obj) in enumerate(items):
            buf_vecs.append(obj.get("embedding") or [])
            buf_meta_src.append((src, obj))
            if len(buf_vecs) >= B:
                _flush()
                if (idx % 50) == 49:
                    await asyncio.sleep(0)
                if len(scored) >= k_eff:
                    break
                if eff_budget is not None and (time.perf_counter() - t0) * 1000.0 > eff_budget:
                    break
        if buf_vecs and len(scored) < k_eff and (eff_budget is None or (time.perf_counter() - t0) * 1000.0 <= eff_budget):
            _flush()

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k_eff]


async def build_context_for(query: str, *, k: int | None = None, max_chars: int = 1500, max_time_ms: int | None = 220) -> str:
    """Build a compact context from top-k snippets (ordered by timestamp)."""
    k = k or DEFAULT_TOP_K
    hits = await retrieve_top_k(query, k=k, max_time_ms=max_time_ms)
    if not hits:
        return ""
    seen: set[str] = set()
    seen_hash: set[str] = set()
    q_hash = hashlib.sha256((query or "").strip().encode("utf-8", errors="ignore")).hexdigest() if query else ""
    parts: List[str] = []
    for score, src, obj in sorted(hits, key=lambda h: float((h[2].get("meta", {}).get("ts") or 0.0))):
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        csha = (meta.get("content_sha256") or "").strip()
        if not pv or pv in seen or is_noise_text(pv) or (q_hash and csha and csha == q_hash):
            continue
        seen.add(pv)
        if csha:
            if csha in seen_hash:
                continue
            seen_hash.add(csha)
        parts.append(pv)
        if sum(len(p) for p in parts) > max_chars:
            break
    if not parts:
        return ""
    body = "\n".join(parts)
    return f"<embeddings_context>\n{body}\n</embeddings_context>"
