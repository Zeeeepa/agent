from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class RTBudget:
    """
    Cooperative real-time budget helper.

    Maintains a soft time budget and yields control to the event loop when the
    budget elapses to reduce scheduling latency for other tasks.
    """

    def __init__(self, budget_ms: int) -> None:
        self._budget_s = max(0.0, budget_ms) / 1000.0
        self._loop = asyncio.get_running_loop()
        self._next = self._loop.time() + self._budget_s

    async def tick(self) -> None:
        now = self._loop.time()
        if now >= self._next:
            await asyncio.sleep(0)
            self._next = self._loop.time() + self._budget_s


@asynccontextmanager
async def rt_section(budget_ms: int = 4) -> AsyncIterator[RTBudget]:
    """
    Async context manager yielding an RTBudget instance.

    Usage:
        async with rt_section(4) as rt:
            for item in items:
                ...
                await rt.tick()
    """
    rt = RTBudget(budget_ms)
    yield rt
