from __future__ import annotations

import asyncio

# Singleton async lock for serialized terminal output across concurrent turns
_PRINT_LOCK: asyncio.Lock | None = None


def get_print_lock() -> asyncio.Lock:
    global _PRINT_LOCK
    if _PRINT_LOCK is None:
        _PRINT_LOCK = asyncio.Lock()
    return _PRINT_LOCK

__all__ = ["get_print_lock"]
