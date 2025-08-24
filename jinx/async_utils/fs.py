from __future__ import annotations

import os
from typing import Optional

import aiofiles
from aiofiles import ospath


async def read_text_raw(path: str) -> str:
    """Read entire text file if it exists else return empty string (no strip)."""
    try:
        if await ospath.exists(path):
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                return await f.read()
        return ""
    except Exception:
        return ""


async def read_text(path: str) -> str:
    """Read entire text file if it exists else return empty string (strip)."""
    txt = await read_text_raw(path)
    return txt.strip() if txt else ""


async def append_line(path: str, text: str) -> None:
    """Append a single line to a log file, creating it if needed."""
    try:
        # Ensure directory exists
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write((text or "") + "\n")
    except Exception:
        # Best-effort semantics
        pass


async def append_and_trim(path: str, text: str, keep_lines: int = 500) -> None:
    """Append text to transcript and trim file to last ``keep_lines`` lines."""
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        lines: list[str]
        if await ospath.exists(path):
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


async def write_text(path: str, text: str) -> None:
    """Overwrite a text file with provided contents (creates parent dirs).

    Best-effort semantics consistent with other helpers: swallow I/O errors.
    """
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(text or "")
    except Exception:
        pass
