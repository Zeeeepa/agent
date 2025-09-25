from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def _run_on_drop(cb: Callable[[], Awaitable[None]]) -> None:
    """Wrap an Awaitable callback into a coroutine for create_task()."""
    await cb()


def try_put_nowait(q: "asyncio.Queue[T]", item: T) -> bool:
    """Attempt to enqueue without blocking. Return True on success.

    Parameters
    ----------
    q : asyncio.Queue[T]
        Target queue.
    item : T
        Item to enqueue.
    """
    try:
        q.put_nowait(item)
        return True
    except asyncio.QueueFull:
        return False


def put_drop_oldest(q: "asyncio.Queue[T]", item: T, on_drop: Callable[[], Awaitable[None]] | None = None) -> None:
    """Enqueue ``item``; if full, drop oldest and log via ``on_drop``.

    Parameters
    ----------
    q : asyncio.Queue[T]
        Target bounded queue.
    item : T
        Item to enqueue.
    on_drop : Optional[Callable[[], Awaitable[None]]]
        Async callback invoked after an item is dropped due to saturation.
    """
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        try:
            _ = q.get_nowait()  # drop oldest
        except asyncio.QueueEmpty:
            pass
        if on_drop:
            # Fire-and-forget; caller may choose to await explicitly instead
            asyncio.create_task(_run_on_drop(on_drop))
        q.put_nowait(item)
