from __future__ import annotations

import asyncio
from typing import Optional

from .crawler import start_realtime_collection
from .pipeline import embed_text


class EmbeddingsService:
    """Micromodule to run realtime embeddings collection in background."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None

    async def _cb(self, text: str, source: str, kind: str) -> None:
        # Best-effort; isolate failures from the tailing loop
        try:
            await embed_text(text, source=source, kind=kind)
        except Exception:
            # Silent drop; logging could be added if needed
            pass

    async def run(self) -> None:
        await start_realtime_collection(self._cb)


def start_embeddings_task() -> asyncio.Task[None]:
    svc = EmbeddingsService()
    return asyncio.create_task(svc.run(), name="embeddings-service")
