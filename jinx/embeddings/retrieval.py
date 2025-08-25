from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import List, Tuple, Dict, Any
import hashlib
import re

from jinx.network_service import get_cortex
from jinx.embeddings.pipeline import iter_recent_items

EMBED_ROOT = os.path.join("log", "embeddings")
DEFAULT_TOP_K = int(os.getenv("EMBED_TOP_K", "5"))
# Balanced defaults; adapt at runtime based on query length
SCORE_THRESHOLD = float(os.getenv("EMBED_SCORE_THRESHOLD", "0.25"))
MIN_PREVIEW_LEN = int(os.getenv("EMBED_MIN_PREVIEW_LEN", "8"))
MAX_FILES_PER_SOURCE = int(os.getenv("EMBED_MAX_FILES_PER_SOURCE", "500"))
MAX_SOURCES = int(os.getenv("EMBED_MAX_SOURCES", "50"))
QUERY_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RECENCY_WINDOW_SEC = int(os.getenv("EMBED_RECENCY_WINDOW_SEC", str(24 * 3600)))


def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return -1.0
    if len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        va = float(a[i])
        vb = float(b[i])
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


_RE_CODEY = re.compile(r"^\s*(print\s*\(|return\b|def\b|class\b|import\b|from\b)", re.I)
_RE_NUMBERY = re.compile(r"^[\s\d+\-*/().]+$")


def _is_noise(pv: str) -> bool:
    # Drop very code-like or purely arithmetic/number lines
    if _RE_CODEY.match(pv):
        return True
    if _RE_NUMBERY.match(pv):
        # ensure it's not just punctuation but includes at least one digit
        return any(ch.isdigit() for ch in pv)
    return False


async def _embed_query(text: str) -> List[float]:
    async def _call():
        return await asyncio.to_thread(
            get_cortex().embeddings.create,
            model=QUERY_MODEL,
            input=text,
        )

    resp = await _call()
    return resp.data[0].embedding if getattr(resp, "data", None) else []


def _iter_items() -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[str, Dict[str, Any]]] = []
    if not os.path.isdir(EMBED_ROOT):
        return items
    # Enumerate sources (directories under EMBED_ROOT, excluding 'index')
    sources = [d for d in os.listdir(EMBED_ROOT) if os.path.isdir(os.path.join(EMBED_ROOT, d)) and d != "index"]
    sources = sources[:MAX_SOURCES]
    for src in sources:
        src_dir = os.path.join(EMBED_ROOT, src)
        try:
            files = [f for f in os.listdir(src_dir) if f.endswith('.json')]
        except FileNotFoundError:
            continue
        # Simple cap to avoid loading too many
        files = files[:MAX_FILES_PER_SOURCE]
        for fn in files:
            p = os.path.join(src_dir, fn)
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                    items.append((src, obj))
            except Exception:
                continue
    return items


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
        if (meta.get("source") or "").strip().lower() != "dialogue":
            continue
        pv = (meta.get("text_preview") or "").strip()
        if len(pv) < MIN_PREVIEW_LEN or _is_noise(pv):
            continue
        sim = _cos(qv, vec)
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
        if src_l != "dialogue" and meta_src_l != "dialogue":
            continue
        pv = (meta.get("text_preview") or "").strip()
        # Filter out very short previews to avoid trivial/noisy lines
        if len(pv) < MIN_PREVIEW_LEN:
            continue
        sim = _cos(qv, vec)
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
    for score, src, obj in hits:
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        csha = (meta.get("content_sha256") or "").strip()
        # Skip if preview empty, duplicate text, or exactly matches the query by hash
        if not pv or pv in seen or (q_hash and csha and csha == q_hash) or _is_noise(pv):
            continue
        seen.add(pv)
        if csha:
            if csha in seen_hash:
                continue
            seen_hash.add(csha)
        role = (meta.get("kind") or "").strip().lower()
        if role not in ("user", "agent"):
            role = "note"
        line = f"{role}: {pv}"
        body_parts.append(line)
        total = sum(len(p) for p in body_parts)
        if total > max_chars:
            break

    if not body_parts:
        return ""

    body = "\n".join(body_parts)
    return f"<embeddings_context>\n{body}\n</embeddings_context>"
