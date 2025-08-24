from __future__ import annotations

from typing import Iterable

import aiofiles
import os


async def append_and_trim(path: str, text: str, keep_lines: int = 500) -> None:
    """Append text to transcript and trim file to last ``keep_lines`` lines.

    Pure, no locking; caller is responsible for synchronization.
    """
    try:
        lines: list[str]
        if os.path.exists(path):
            try:
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    content = await f.read()
                lines = content.splitlines()
            except FileNotFoundError:
                lines = []
        else:
            lines = []
        lines = lines + ["", text]
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write("\n".join(lines[-keep_lines:]) + "\n")
    except Exception:
        # Best-effort; swallow I/O errors to mirror existing semantics
        pass
