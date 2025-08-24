from __future__ import annotations

from jinx.async_utils.fs import append_line as _append_line


async def append_line(path: str, text: str) -> None:
    """Append a single line to a log file, creating it if needed.

    Delegates to `jinx.async_utils.fs.append_line` to avoid duplication.
    """
    await _append_line(path, text)
