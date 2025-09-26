from __future__ import annotations

import asyncio
import heapq
import contextlib
from typing import List, Tuple

from jinx.settings import Settings


def classify_priority(msg: str) -> int:
    """
    Classify a message priority.

    Lower numbers are higher priority.
    0 = high (e.g., commands starting with '!'), 1 = normal, 2 = low (e.g., '<no_response>').
    """
    s = msg.strip()
    if not s:
        return 1
    if s.startswith("!"):
        return 0
    if s == "<no_response>":
        return 2
    return 1


def start_priority_dispatcher_task(src: "asyncio.Queue[str]", dst: "asyncio.Queue[str]", settings: Settings) -> "asyncio.Task[None]":
    async def _run() -> None:
        # Priority reordering without changing downstream API types.
        # We buffer in a local heap and forward to dst as capacity allows.
        loop = asyncio.get_running_loop()
        budget = max(1, settings.runtime.hard_rt_budget_ms) / 1000.0
        next_yield = loop.time() + budget

        heap: List[Tuple[int, int, str]] = []
        seq = 0

        async def _awaitable_src_get() -> str:
            return await src.get()

        while True:
            if not settings.runtime.use_priority_queue:
                # Fast path: FIFO pass-through with cooperative yield
                msg = await src.get()
                await dst.put(msg)
                if loop.time() >= next_yield:
                    await asyncio.sleep(0)
                    next_yield = loop.time() + budget
                continue

            # Priority mode: race new item vs flush tick
            get_task = asyncio.create_task(_awaitable_src_get())
            flush_task = asyncio.create_task(asyncio.sleep(0))
            try:
                done, _ = await asyncio.wait({get_task, flush_task}, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                get_task.cancel(); flush_task.cancel()
                raise

            if get_task in done:
                try:
                    msg = get_task.result()
                except asyncio.CancelledError:
                    raise
                pr = classify_priority(msg)
                heapq.heappush(heap, (pr, seq, msg))
                seq += 1
            # Cancel pending tasks
            if not get_task.done():
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
            if not flush_task.done():
                flush_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await flush_task

            if heap:
                _, _, item = heapq.heappop(heap)
                await dst.put(item)

            if loop.time() >= next_yield:
                await asyncio.sleep(0)
                next_yield = loop.time() + budget

    return asyncio.create_task(_run())
