from __future__ import annotations

"""
Centralized CPU threadpool for RT-friendly offloading.

Use run_cpu(func, *args, **kwargs) to execute CPU-bound work off the event loop.
Max workers are controlled by env JINX_CPU_WORKERS (default: min(8, max(2, cpu_count))).
"""

import os
import concurrent.futures as _fut
from functools import partial
from typing import Any, Callable
import asyncio


def _calc_workers() -> int:
    try:
        v = int(os.getenv("JINX_CPU_WORKERS", "0"))
        if v > 0:
            return v
    except Exception:
        pass
    try:
        import os as _os
        c = _os.cpu_count() or 4
    except Exception:
        c = 4
    return max(2, min(8, int(c)))


_CPU_EXECUTOR: _fut.ThreadPoolExecutor | None = None


def _get_exec() -> _fut.ThreadPoolExecutor:
    global _CPU_EXECUTOR
    if _CPU_EXECUTOR is None:
        _CPU_EXECUTOR = _fut.ThreadPoolExecutor(max_workers=_calc_workers(), thread_name_prefix="jinx-cpu")
    return _CPU_EXECUTOR


async def run_cpu(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    execu = _get_exec()
    return await loop.run_in_executor(execu, partial(func, *args, **kwargs))
