from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from typing import Dict, Any, Deque, Iterable

from .paths import EMBED_ROOT, ensure_dirs
from .util import sha256_text, now_ts
from .text_clean import strip_known_tags, is_noise_text
from .index_io import append_index
from .embed_cache import embed_text_cached
from .project_terms import extract_terms as _extract_terms
from .fingerprint import simhash as _simhash
from jinx.micro.text.heuristics import is_code_like as _is_code_like
_RECENT_MAX = 200
_recent: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_MAX)


async def embed_text(text: str, *, source: str = "default", persist: bool = True, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """Create an embedding for text and persist versioned artifact.

    Storage layout:
    - log/embeddings/{source}/{hash}.json  -> embedding item
    - log/embeddings/index/{source}.jsonl  -> append-only index
    """
    ensure_dirs()
    raw = (text or "").strip()
    if not raw:
        return {"skipped": True, "reason": "empty"}
    cleaned = strip_known_tags(raw)
    text = cleaned.strip()
    if not text:
        return {"skipped": True, "reason": "empty_after_tags"}
    if is_noise_text(text):
        return {"skipped": True, "reason": "empty"}

    content_id = sha256_text(text)
    source_dir = os.path.join(EMBED_ROOT, source)
    os.makedirs(source_dir, exist_ok=True)

    item_path = os.path.join(source_dir, f"{content_id}.json")
    if os.path.exists(item_path):
        # Already embedded; still record a touch in index
        try:
            def _read_cached() -> dict | None:
                try:
                    with open(item_path, "r", encoding="utf-8") as r:
                        return json.load(r)
                except Exception:
                    return None
            cached_obj = await asyncio.to_thread(_read_cached)
        except Exception:
            cached_obj = None
        else:
            await append_index(source, {
                "ts": now_ts(),
                "source": source,
                "kind": kind,
                "content_id": content_id,
                "dedup": True,
            })
            # Also surface to recent cache for real-time retrieval
            try:
                if cached_obj:
                    _recent.appendleft(cached_obj)
            except Exception:
                pass
            return {"cached": True, **(cached_obj or {})}

    # Call embeddings through shared cached helper (TTL, coalescing, limits, timeout)
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    try:
        vec = await embed_text_cached(text, model=model)
    except Exception:
        vec = []

    # Enrich metadata with lightweight features
    try:
        terms = _extract_terms(text)[:24]
    except Exception:
        terms = []
    try:
        fp = _simhash(text)
    except Exception:
        fp = 0
    try:
        is_code = bool(_is_code_like(text))
    except Exception:
        is_code = False
    # Prefer per-turn group id over env session
    try:
        _sess = (_get_group() or "").strip()
        session = _sess or None
    except Exception:
        session = None
    if session is None:
        session = (os.getenv("JINX_SESSION", "").strip() or None)

    meta: Dict[str, Any] = {
        "ts": now_ts(),
        "model": model,
        "source": source,
        "kind": kind,
        "content_sha256": content_id,
        "dims": len(vec) if vec is not None else 0,
        "text_preview": text[:256],
        "terms": terms,
        "fingerprint": fp,
        "is_code": is_code,
        **({"session": session} if session else {}),
    }

    payload = {
        "meta": meta,
        "embedding": vec,
    }

    def _write_payload() -> None:
        with open(item_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    try:
        await asyncio.to_thread(_write_payload)
    except Exception:
        pass

    await append_index(source, {
        "ts": meta["ts"],
        "source": source,
        "kind": kind,
        "content_id": content_id,
    })

    # Push to in-memory recent cache for real-time use
    try:
        _recent.appendleft(payload)
    except Exception:
        pass

    return payload


def iter_recent_items() -> Iterable[Dict[str, Any]]:
    """Return a snapshot iterator over recent embedded payloads (most recent first)."""
    return list(_recent)
