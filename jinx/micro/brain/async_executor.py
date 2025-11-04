"""Async Executor - thread pool для CPU-bound и I/O операций.

Предотвращает блокировку event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional


class AsyncExecutor:
    """Thread pool executor для неблокирующих операций."""
    
    _instance: Optional['AsyncExecutor'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="jinx-brain-worker"
        )
    
    @classmethod
    async def get_instance(cls) -> 'AsyncExecutor':
        """Get singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = AsyncExecutor()
        return cls._instance
    
    async def run(self, func: Callable, *args, **kwargs) -> Any:
        """Run blocking function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: func(*args, **kwargs)
        )
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor."""
        try:
            self.executor.shutdown(wait=wait)
        except Exception:
            pass


# Global helper functions
async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """Run blocking function in thread pool (global helper)."""
    executor = await AsyncExecutor.get_instance()
    return await executor.run(func, *args, **kwargs)


async def run_cpu_bound(func: Callable, *args, **kwargs) -> Any:
    """Run CPU-bound function in thread pool."""
    return await run_in_thread(func, *args, **kwargs)


__all__ = [
    "AsyncExecutor",
    "run_in_thread",
    "run_cpu_bound",
]
