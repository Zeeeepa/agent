from __future__ import annotations

import asyncio
import time
import os
from typing import List, Tuple, Dict, Any
import hashlib

from jinx.net import get_openai_client
from jinx.micro.embeddings.pipeline import iter_recent_items
from .paths import EMBED_ROOT
from .util import cos
from .text_clean import is_noise_text
from .scan_store import iter_items as scan_iter_items

DEFAULT_TOP_K = int(os.getenv("EMBED_TOP_K", "5"))
# Balanced defaults; adapt at runtime based on query length
SCORE_THRESHOLD = float(os.getenv("EMBED_SCORE_THRESHOLD", "0.25"))
MIN_PREVIEW_LEN = int(os.getenv("EMBED_MIN_PREVIEW_LEN", "8"))
MAX_FILES_PER_SOURCE = int(os.getenv("EMBED_MAX_FILES_PER_SOURCE", "500"))
MAX_SOURCES = int(os.getenv("EMBED_MAX_SOURCES", "50"))
QUERY_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RECENCY_WINDOW_SEC = int(os.getenv("EMBED_RECENCY_WINDOW_SEC", str(24 * 3600)))


async def _embed_query(text: str) -> List[float]:
    async def _call():
        def _worker():
            client = get_openai_client()
            return client.embeddings.create(model=QUERY_MODEL, input=text)
        return await asyncio.to_thread(_worker)

    try:
        resp = await _call()
        return resp.data[0].embedding if getattr(resp, "data", None) else []
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

    qv = await _embed_query(query)
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    now = time.time()
    t0 = time.perf_counter()

    # 1) Fast-path: score in-memory recent items first
    for obj in iter_recent_items():
        vec = obj.get("embedding") or []
        meta = obj.get("meta", {})
        src_l = (meta.get("source") or "").strip().lower()
        if not (src_l == "dialogue" or src_l.startswith("sandbox/")):
            continue
        pv = (meta.get("text_preview") or "").strip()
        if len(pv) < MIN_PREVIEW_LEN or is_noise_text(pv):
            continue
        sim = cos(qv, vec)
        if sim < thr:
            continue
        ts = float(meta.get("ts") or 0.0)
        age = max(0.0, now - ts)
        rec = 0.0 if RECENCY_WINDOW_SEC <= 0 else max(0.0, 1.0 - (age / RECENCY_WINDOW_SEC))
        score = 0.8 * sim + 0.2 * rec
        scored.append((score, meta.get("source", "recent"), obj))
        if len(scored) >= k_eff:
            break

    # Early return if we already have enough and time budget is tight
    if len(scored) >= k_eff:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k_eff]
    if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k_eff]

    # 2) Fallback: scan persisted files (bounded and time-guarded)
    for src, obj in _iter_items():
        vec = obj.get("embedding") or []
        meta = obj.get("meta", {})
        src_l = (src or "").strip().lower()
        meta_src_l = (meta.get("source") or "").strip().lower()
        allow_src = (
            src_l == "dialogue" or src_l.startswith("sandbox/") or
            meta_src_l == "dialogue" or meta_src_l.startswith("sandbox/")
        )
        if not allow_src:
            continue
        pv = (meta.get("text_preview") or "").strip()
        # Filter out very short previews to avoid trivial/noisy lines
        if len(pv) < MIN_PREVIEW_LEN:
            continue
        sim = cos(qv, vec)
        if sim < thr:
            continue
        ts = float(meta.get("ts") or 0.0)
        age = max(0.0, now - ts)
        rec = 0.0 if RECENCY_WINDOW_SEC <= 0 else max(0.0, 1.0 - (age / RECENCY_WINDOW_SEC))
        score = 0.8 * sim + 0.2 * rec
        scored.append((score, src, obj))
        if len(scored) >= k_eff:
            break
        if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            break
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k_eff]


async def build_context_for(query: str, *, k: int | None = None, max_chars: int = 1500, max_time_ms: int | None = 220) -> str:
    """Build a context string from top-k similar snippets.

    Pulls `text_preview` from stored metadata to remain compact.
    """
    k = k or DEFAULT_TOP_K
    hits = await retrieve_top_k(query, k=k, max_time_ms=max_time_ms)
    if not hits:
        return ""
    # Deduplicate identical previews and identical content by hash while keeping order
    seen: set[str] = set()
    seen_hash: set[str] = set()
    q_hash = hashlib.sha256((query or "").strip().encode("utf-8", errors="ignore")).hexdigest() if query else ""
    body_parts: List[str] = []
    # Preserve chronological order in the final context to avoid semantic chaos.
    # We first select by similarity (retrieve_top_k), then sort chosen items by their timestamp.
    hits_sorted = sorted(
        hits,
        key=lambda h: float((h[2].get("meta", {}).get("ts") or 0.0)),
    )
    for score, src, obj in hits_sorted:
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        csha = (meta.get("content_sha256") or "").strip()
        # Skip if preview empty, duplicate text, or exactly matches the query by hash
        if not pv or pv in seen or (q_hash and csha and csha == q_hash) or is_noise_text(pv):
            continue
        seen.add(pv)
        if csha:
            if csha in seen_hash:
                continue
            seen_hash.add(csha)
        # Keep original line breaks for readability; collapse only excessive trailing/leading space
        if not pv:
            continue
        body_parts.append(pv)
        total = sum(len(p) for p in body_parts)
        if total > max_chars:
            break

    if not body_parts:
        return ""

    # Join with a blank line between hints for readability and add padding inside the tag
    body = "\n".join(body_parts)
    return f"<embeddings_context>\n{body}\n</embeddings_context>"
