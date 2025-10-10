from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

_ENABLED = None

def _on() -> bool:
    global _ENABLED
    if _ENABLED is None:
        try:
            _ENABLED = (os.getenv("JINX_TIMING", "0").strip().lower() not in ("", "0", "false", "off", "no"))
        except Exception:
            _ENABLED = False
    return bool(_ENABLED)


@asynccontextmanager
async def timing_section(name: str) -> AsyncIterator[None]:
    if not _on():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        try:
            from jinx.logger.file_logger import append_line as _append
            from jinx.log_paths import BLUE_WHISPERS
            await _append(BLUE_WHISPERS, f"[timing] {name} {dt:.1f}ms")
        except Exception:
            pass
