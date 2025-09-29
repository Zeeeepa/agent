from __future__ import annotations

import os
import asyncio
from typing import Optional

from jinx.async_utils.fs import read_text_raw


def _truthy(name: str, default: str = "1") -> bool:
    try:
        return str(os.getenv(name, default)).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


async def _get_size_bytes(path: str) -> int:
    try:
        st = await asyncio.to_thread(os.stat, path)
        return int(st.st_size)
    except Exception:
        return 0


async def _get_line_count(path: str) -> int:
    try:
        txt = await read_text_raw(path)
        if txt == "":
            return 0
        return len(txt.splitlines())
    except Exception:
        return 0


async def maybe_warn_filesize(path: str) -> Optional[str]:
    """Return a warning string if file exceeds configured thresholds, else None.

    Controls:
    - JINX_FILESIZE_WARN: enable/disable
    - JINX_FILESIZE_MAXLINES: default 1200
    - JINX_FILESIZE_MAXBYTES: default 150000 (approx 150KB)
    """
    if not _truthy("JINX_FILESIZE_WARN", "1"):
        return None
    try:
        max_lines = int(os.getenv("JINX_FILESIZE_MAXLINES", "1200"))
    except Exception:
        max_lines = 1200
    try:
        max_bytes = int(os.getenv("JINX_FILESIZE_MAXBYTES", "150000"))
    except Exception:
        max_bytes = 150000
    try:
        b = await _get_size_bytes(path)
        n = await _get_line_count(path)
        msgs = []
        if max_lines > 0 and n > max_lines:
            msgs.append(f"lines={n} > max_lines={max_lines}")
        if max_bytes > 0 and b > max_bytes:
            msgs.append(f"bytes={b} > max_bytes={max_bytes}")
        if msgs:
            return f"watchdog: large file '{path}': " + ", ".join(msgs)
    except Exception:
        return None
    return None
