"""Debug Logger - Centralized debug logging to file.

Provides non-blocking async logging for debug messages without cluttering terminal.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Optional


class DebugLogger:
    """Async debug logger writing to file."""
    
    def __init__(self, log_path: str = "log/debug.log"):
        self._log_path = log_path
        self._lock = asyncio.Lock()
        self._initialized = False
        self._max_size = 10 * 1024 * 1024  # 10MB
    
    async def _ensure_initialized(self):
        """Ensure log directory exists."""
        if self._initialized:
            return
        
        try:
            log_dir = os.path.dirname(self._log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._initialized = True
        except Exception:
            pass
    
    async def log(self, message: str, category: str = "DEBUG"):
        """Log a debug message to file."""
        try:
            await self._ensure_initialized()
            
            # Rotate if too large
            try:
                if os.path.exists(self._log_path):
                    size = os.path.getsize(self._log_path)
                    if size > self._max_size:
                        # Rotate: keep last 50%
                        try:
                            with open(self._log_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                            
                            keep = lines[len(lines)//2:]
                            
                            with open(self._log_path, 'w', encoding='utf-8') as f:
                                f.writelines(keep)
                        except Exception:
                            pass
            except Exception:
                pass
            
            # Format message
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            formatted = f"[{timestamp}] [{category}] {message}\n"
            
            # Write async
            async with self._lock:
                try:
                    # Use blocking I/O in executor for safety
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        self._write_sync,
                        formatted
                    )
                except Exception:
                    pass
        
        except Exception:
            pass  # Silent fail for debug logger
    
    def _write_sync(self, text: str):
        """Synchronous write helper."""
        try:
            with open(self._log_path, 'a', encoding='utf-8') as f:
                f.write(text)
        except Exception:
            pass


# Singleton instance
_logger: Optional[DebugLogger] = None
_logger_lock = asyncio.Lock()


async def get_debug_logger() -> DebugLogger:
    """Get singleton debug logger."""
    global _logger
    if _logger is None:
        async with _logger_lock:
            if _logger is None:
                _logger = DebugLogger()
    return _logger


async def debug_log(message: str, category: str = "DEBUG"):
    """Log debug message to file."""
    try:
        logger = await get_debug_logger()
        await logger.log(message, category)
    except Exception:
        pass  # Silent fail


def debug_log_sync(message: str, category: str = "DEBUG"):
    """Synchronous debug log (creates task)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(debug_log(message, category))
        else:
            # If no loop, write directly
            logger = DebugLogger()
            logger._write_sync(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{category}] {message}\n")
    except Exception:
        pass


__all__ = [
    "DebugLogger",
    "get_debug_logger",
    "debug_log",
    "debug_log_sync",
]
