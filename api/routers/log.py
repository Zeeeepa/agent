from __future__ import annotations

from fastapi import APIRouter, HTTPException
from ..schemas.log import Log
from ..services.log_service import LogService

router = APIRouter()
svc = LogService()

@router.get('/')
async def list_log() -> list[Log]:
    return await svc.list()

@router.get('/{id}')
async def get_log(id: int) -> Log:
    obj = await svc.get(id)
    if not obj:
        raise HTTPException(status_code=404, detail='Log not found')
    return obj

@router.post('/')
async def create_log(payload: Log) -> Log:
    return await svc.create(payload)

@router.put('/{id}')
async def update_log(id: int, payload: Log) -> Log:
    obj = await svc.update(id, payload)
    if not obj:
        raise HTTPException(status_code=404, detail='Log not found')
    return obj

@router.delete('/{id}')
async def delete_log(id: int) -> dict:
    ok = await svc.delete(id)
    if not ok:
        raise HTTPException(status_code=404, detail='Log not found')
    return {'ok': True}
