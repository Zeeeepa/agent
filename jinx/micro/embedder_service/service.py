from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from jinx.micro.embeddings.embed_cache import (
    embed_text_cached as _embed_one_cached,
    embed_texts_cached as _embed_many_cached,
)


class EmbedderService:
    """Embedder Service (micro-module).

    Thin async wrapper around embedding backends with a stable contract.
    Heavy logic for model management should live here; facades re-export.

    Contract (initial skeleton):
    - embed_one(text, model_version?) -> vector
    - embed(texts, model_version?) -> List[vector]
    - update_model(data) -> bool  (stub)
    - similarity_query(qv, k, index_version?) -> List[...]  (stub)
    """

    def __init__(self, *, default_model: Optional[str] = None) -> None:
        self.default_model = default_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    async def embed_one(self, text: str, model_version: Optional[str] = None) -> List[float]:
        model = (model_version or self.default_model)
        try:
            return await _embed_one_cached(text or "", model=model)
        except Exception:
            return []

    async def embed(self, texts: List[str], model_version: Optional[str] = None) -> List[List[float]]:
        model = (model_version or self.default_model)
        try:
            return await _embed_many_cached(list(texts or []), model=model)
        except Exception:
            return [[] for _ in (texts or [])]

    async def update_model(self, data: Dict[str, Any]) -> bool:  # skeleton
        # Placeholder: future fine-tune / version switch logic
        _ = data
        return False

    async def similarity_query(self, qv: List[float], k: int, *, index_version: Optional[str] = None) -> List[Dict[str, Any]]:  # skeleton
        # Intentionally left blank in skeleton; memory_service will own indices.
        _ = (qv, k, index_version)
        return []


async def start_embedder_service_task(*, default_model: Optional[str] = None) -> asyncio.Task[None]:
    svc = EmbedderService(default_model=default_model)

    async def _noop_forever() -> None:
        # Skeleton doesn't host a network server yet; keep a cancellable task alive.
        while True:
            await asyncio.sleep(3600)

    return asyncio.create_task(_noop_forever(), name="embedder-service")
