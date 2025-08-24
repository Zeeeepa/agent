from __future__ import annotations

from jinx.async_utils.fs import read_text_raw


async def read_transcript(path: str) -> str:
    """Return the contents of the transcript file or empty string on error.

    Pure function: no locking; caller is responsible for synchronization.
    """
    try:
        return await read_text_raw(path)
    except Exception:
        return ""
