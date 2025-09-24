from __future__ import annotations

from typing import TypedDict


class Chunk(TypedDict):
    text: str
    line_start: int
    line_end: int
