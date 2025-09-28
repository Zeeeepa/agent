from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List

Handler = Callable[[str, Any], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Handler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, handler: Handler) -> None:
        async with self._lock:
            self._subs[topic].append(handler)

    async def publish(self, topic: str, payload: Any) -> None:
        # fan-out asynchronously; do not await all to finish in-line
        handlers: List[Handler]
        async with self._lock:
            handlers = list(self._subs.get(topic, ()))
        for h in handlers:
            try:
                # schedule; let handlers manage their own budgets
                asyncio.create_task(h(topic, payload))
            except Exception:
                # swallow — bus must never raise
                pass


_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
