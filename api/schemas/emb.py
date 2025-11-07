from __future__ import annotations

from pydantic import BaseModel

class Emb(BaseModel):
    id: int
    emb_name: str | None = None
