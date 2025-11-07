from __future__ import annotations

from ..repositories.log_repo import LogRepository
from ..schemas.log import Log

class LogService:
    def __init__(self):
        self.repo = LogRepository()

    async def list(self) -> list[Log]:
        return await self.repo.list()

    async def get(self, id: int) -> Log | None:
        return await self.repo.get(id)

    async def create(self, obj: Log) -> Log:
        return await self.repo.create(obj)

    async def update(self, id: int, obj: Log) -> Log | None:
        return await self.repo.update(id, obj)

    async def delete(self, id: int) -> bool:
        return await self.repo.delete(id)
