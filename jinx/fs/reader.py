from __future__ import annotations

import aiofiles
from aiofiles import ospath


async def read_text(path: str) -> str:
    """Read entire text file if it exists else return empty string."""
    try:
        if await ospath.exists(path):
            async with aiofiles.open(path, encoding="utf-8") as f:
                return (await f.read()).strip()
        return ""
    except Exception:
        # Maintain silent failure semantics similar to existing wire()
        return ""
