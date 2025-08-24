from __future__ import annotations

import aiofiles


async def append_line(path: str, text: str) -> None:
    """Append a single line to a log file, creating it if needed.

    Pure function; caller ensures any needed synchronization.
    """
    try:
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write((text or "") + "\n")
    except Exception:
        # Mirror existing best-effort semantics
        pass
