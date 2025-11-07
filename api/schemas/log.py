from __future__ import annotations

from pydantic import BaseModel

class Log(BaseModel):
    id: int
    log_name: str | None = None
