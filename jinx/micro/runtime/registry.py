from __future__ import annotations

import asyncio
from typing import Dict, Optional


class _Registry:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._items: Dict[str, object] = {}

    async def put(self, pid: str, prog: object) -> None:
        async with self._lock:
            self._items[pid] = prog

    async def get(self, pid: str) -> Optional[object]:
        async with self._lock:
            return self._items.get(pid)

    async def remove(self, pid: str) -> None:
        async with self._lock:
            self._items.pop(pid, None)

    async def list_ids(self) -> list[str]:
        async with self._lock:
            return list(self._items.keys())


_registry: _Registry | None = None


def get_registry() -> _Registry:
    global _registry
    if _registry is None:
        _registry = _Registry()
    return _registry
