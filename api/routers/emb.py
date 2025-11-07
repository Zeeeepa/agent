from __future__ import annotations

from fastapi import APIRouter, HTTPException
from ..schemas.emb import Emb
from ..services.emb_service import EmbService

router = APIRouter()
svc = EmbService()

@router.get('/')
async def list_emb() -> list[Emb]:
    return await svc.list()

@router.get('/{id}')
async def get_emb(id: int) -> Emb:
    obj = await svc.get(id)
    if not obj:
        raise HTTPException(status_code=404, detail='Emb not found')
    return obj

@router.post('/')
async def create_emb(payload: Emb) -> Emb:
    return await svc.create(payload)

@router.put('/{id}')
async def update_emb(id: int, payload: Emb) -> Emb:
    obj = await svc.update(id, payload)
    if not obj:
        raise HTTPException(status_code=404, detail='Emb not found')
    return obj

@router.delete('/{id}')
async def delete_emb(id: int) -> dict:
    ok = await svc.delete(id)
    if not ok:
        raise HTTPException(status_code=404, detail='Emb not found')
    return {'ok': True}
