from __future__ import annotations

import os
from typing import Optional, TypeVar, Callable, Any
from functools import wraps
import hashlib

import aiofiles
from aiofiles import ospath
import asyncio as _asyncio
from collections import OrderedDict as _OD
import os as _os

T = TypeVar('T')


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
    Invalidates cache entries for the written file.
    """
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(text or "")
        
        # Invalidate cache entries for this path
        global _read_lru_total_size
        async with _read_lru_lock:
            keys_to_remove = [k for k in _read_lru.keys() if k[0] == path]
            for key in keys_to_remove:
                entry = _read_lru.pop(key, None)
                if entry:
                    _read_lru_total_size -= entry.size
    except Exception:
        pass


# --- Advanced LRU cache with TTL and size limits ---
_READ_LRU_CAP = 256  # Increased capacity for better hit rates
_READ_LRU_TTL_S = 300  # 5 minute TTL for cache entries
_READ_LRU_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB max total cache size

class _LRUEntry:
    __slots__ = ('content', 'size', 'timestamp')
    def __init__(self, content: str, size: int, timestamp: float):
        self.content = content
        self.size = size
        self.timestamp = timestamp

_read_lru: _OD[tuple[str, int, int], _LRUEntry] = _OD()
_read_lru_total_size: int = 0
_read_lru_lock = _asyncio.Lock()


def _read_abs_sync(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


async def read_text_abs_thread(path: str) -> str:
    """Read absolute file text via thread with advanced LRU cache (TTL + size limits).

    Cache features:
    - LRU eviction policy
    - TTL-based expiration (5 min)
    - Total size limit (10MB)
    - Async-safe with lock protection
    
    Returns empty string on error.
    """
    global _read_lru_total_size
    
    try:
        st = _os.stat(path)
        key = (path, int(st.st_mtime), int(st.st_size))
        file_size = int(st.st_size)
    except Exception:
        key = (path, 0, 0)
        file_size = 0
    
    async with _read_lru_lock:
        # Check cache with TTL validation
        if key in _read_lru:
            entry = _read_lru[key]
            import time
            if (time.time() - entry.timestamp) < _READ_LRU_TTL_S:
                # move to end (most recent)
                _read_lru.move_to_end(key)
                return entry.content
            else:
                # Expired entry, remove it
                _read_lru_total_size -= entry.size
                del _read_lru[key]
    
    # Cache miss - read from disk
    txt = await _asyncio.to_thread(_read_abs_sync, path)
    txt_size = len(txt.encode('utf-8', errors='ignore'))
    
    async with _read_lru_lock:
        import time
        entry = _LRUEntry(txt, txt_size, time.time())
        
        # Evict entries if adding this would exceed size limit
        while _read_lru and (_read_lru_total_size + txt_size) > _READ_LRU_MAX_SIZE_BYTES:
            oldest_key, oldest_entry = _read_lru.popitem(last=False)
            _read_lru_total_size -= oldest_entry.size
        
        # Add new entry
        _read_lru[key] = entry
        _read_lru_total_size += txt_size
        
        # Also enforce count limit
        while len(_read_lru) > _READ_LRU_CAP:
            oldest_key, oldest_entry = _read_lru.popitem(last=False)
            _read_lru_total_size -= oldest_entry.size
    
    return txt
