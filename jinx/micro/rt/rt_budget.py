from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional


def env_ms(name: str, default_ms: int) -> int:
    try:
        import os
        v = int(str(getattr(__import__('os'), 'getenv')(name, str(default_ms))))
        return max(0, v)
    except Exception:
        return int(default_ms)


async def run_bounded(awaitable: Awaitable[Any], timeout_ms: int, *, on_timeout: Optional[Callable[[], None]] = None, cancel: bool = True) -> Any | None:
    """Await an awaitable with a time budget in milliseconds.

    - Returns the result on success.
    - Returns None on timeout (after optional cancellation and on_timeout callback).
    - Raises other exceptions from the awaitable as-is.
    """
    if timeout_ms is None or timeout_ms <= 0:
        return await awaitable
    timeout_s = max(0.001, float(timeout_ms) / 1000.0)
    # Always operate on a Task to avoid reusing a coroutine object
    task: asyncio.Task
    if isinstance(awaitable, asyncio.Task):
        task = awaitable
    else:
        loop = asyncio.get_running_loop()
        task = loop.create_task(awaitable)  # type: ignore[arg-type]
    # Wait without implicit cancellation to respect 'cancel' flag
    done, _ = await asyncio.wait({task}, timeout=timeout_s)
    if task in done:
        # Propagate result or exception
        return await task
    # Timeout
    try:
        if cancel and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        if on_timeout:
            try:
                on_timeout()
            except Exception:
                pass
    return None
