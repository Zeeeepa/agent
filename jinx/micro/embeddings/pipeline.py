from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from typing import Dict, Any, Deque, Iterable

from jinx.net import get_openai_client
from .paths import EMBED_ROOT, ensure_dirs
from .util import sha256_text, now_ts
from .text_clean import strip_known_tags, is_noise_text
from .index_io import append_index

_RECENT_MAX = 200
_recent: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_MAX)


async def embed_text(text: str, *, source: str, kind: str = "text") -> Dict[str, Any]:
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

    # Call OpenAI embeddings through existing client; use to_thread for sync SDK
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    async def _call() -> Any:
        def _worker() -> Any:
            client = get_openai_client()
            return client.embeddings.create(model=model, input=text)
        return await asyncio.to_thread(_worker)

    try:
        resp = await _call()
        vec = resp.data[0].embedding if getattr(resp, "data", None) else []
    except Exception:
        # Best-effort: fallback to empty vector on API failure
        vec = []

    meta: Dict[str, Any] = {
        "ts": now_ts(),
        "model": model,
        "source": source,
        "kind": kind,
        "content_sha256": content_id,
        "dims": len(vec) if vec is not None else 0,
        "text_preview": text[:256],
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
