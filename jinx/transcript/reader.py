from __future__ import annotations

import aiofiles


async def read_transcript(path: str) -> str:
    """Return the contents of the transcript file or empty string on error.

    Pure function: no locking; caller is responsible for synchronization.
    """
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()
    except Exception:
        return ""
