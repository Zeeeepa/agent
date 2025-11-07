from __future__ import annotations

from ..repositories.emb_repo import EmbRepository
from ..schemas.emb import Emb

class EmbService:
    def __init__(self):
        self.repo = EmbRepository()

    async def list(self) -> list[Emb]:
        return await self.repo.list()

    async def get(self, id: int) -> Emb | None:
        return await self.repo.get(id)

    async def create(self, obj: Emb) -> Emb:
        return await self.repo.create(obj)

    async def update(self, id: int, obj: Emb) -> Emb | None:
        return await self.repo.update(id, obj)

    async def delete(self, id: int) -> bool:
        return await self.repo.delete(id)
