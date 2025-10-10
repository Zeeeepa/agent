from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Tuple

# Generic hot-snapshot cache for expensive on-disk scans.
# - Keeps last snapshot in-memory.
# - Refreshes in background after TTL while serving the last snapshot immediately.
# - First call awaits initial load to avoid returning empty.
# - Keyed by an arbitrary string key so multiple stores can coexist (runtime, project, etc.).

_SNAPSHOTS: Dict[str, Tuple[List[Any], float]] = {}
_TASKS: Dict[str, asyncio.Task] = {}
_LOCKS: Dict[str, asyncio.Lock] = {}


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


async def get_hot_snapshot(key: str, loader: Callable[[], Awaitable[List[Any]]], ttl_ms: int) -> List[Any]:
    ttl_ms = max(0, int(ttl_ms))
    snap, ts = _SNAPSHOTS.get(key, ([], 0.0))
    # Serve fresh snapshot if within TTL
    if snap and ttl_ms > 0 and (_now_ms() - ts) <= ttl_ms:
        return snap
    # If a refresh is running, serve current snapshot without blocking
    t = _TASKS.get(key)
    if t is not None and not t.done():
        return snap
    # Lock per key to avoid double refresh
    lock = _LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        snap, ts = _SNAPSHOTS.get(key, (snap, ts))
        if snap and ttl_ms > 0 and (_now_ms() - ts) <= ttl_ms:
            return snap
        # Start loader task
        task = asyncio.create_task(loader())
        _TASKS[key] = task
        # If we have no snapshot yet, await first load; else refresh in background
        if not snap:
            try:
                res = await task
            except Exception:
                res = []
            _SNAPSHOTS[key] = (res or [], _now_ms())
            return _SNAPSHOTS[key][0]

        def _on_done(t: asyncio.Task) -> None:
            try:
                res = t.result()
            except Exception:
                return
            _SNAPSHOTS[key] = (res or [], _now_ms())

        task.add_done_callback(_on_done)
        return snap


# Convenience wrappers
async def get_runtime_items_hot(loader: Callable[[], Awaitable[List[Any]]], ttl_ms: int) -> List[Any]:
    return await get_hot_snapshot("runtime_items", loader, ttl_ms)


async def get_project_chunks_hot(loader: Callable[[], Awaitable[List[Any]]], ttl_ms: int) -> List[Any]:
    return await get_hot_snapshot("project_chunks", loader, ttl_ms)
