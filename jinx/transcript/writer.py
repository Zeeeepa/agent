from __future__ import annotations

from typing import Iterable
from jinx.async_utils.fs import append_and_trim as _append_and_trim


async def append_and_trim(path: str, text: str, keep_lines: int = 500) -> None:
    """Append text to transcript and trim file to last ``keep_lines`` lines.

    Delegates to `jinx.async_utils.fs.append_and_trim` to avoid duplication.
    """
    await _append_and_trim(path, text, keep_lines)
