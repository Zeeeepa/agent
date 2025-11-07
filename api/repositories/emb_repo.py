from __future__ import annotations

import asyncio
from ..schemas.emb import Emb

class EmbRepository:
    def __init__(self):
        self._data: dict[int, Emb] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def list(self) -> list[Emb]:
        async with self._lock:
            return list(self._data.values())

    async def get(self, id: int) -> Emb | None:
        async with self._lock:
            return self._data.get(id)

    async def create(self, obj: Emb) -> Emb:
        async with self._lock:
            oid = self._next_id; self._next_id += 1
            obj.id = oid
            self._data[oid] = obj
            return obj

    async def update(self, id: int, obj: Emb) -> Emb | None:
        async with self._lock:
            if id not in self._data:
                return None
            obj.id = id
            self._data[id] = obj
            return obj

    async def delete(self, id: int) -> bool:
        async with self._lock:
            return self._data.pop(id, None) is not None
