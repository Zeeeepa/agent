"""Advanced real-time budget management with deadline tracking."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T')


class DeadlineStatus(Enum):
    """Status of deadline-based execution."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class BudgetResult:
    """Result of budget-constrained execution."""
    status: DeadlineStatus
    value: Any = None
    elapsed_ms: float = 0.0
    exception: Optional[Exception] = None


def env_ms(name: str, default_ms: int) -> int:
    """Parse millisecond value from environment with validation.
    
    This is a performance-critical hot path function, optimized for speed.
    """
    try:
        import os
        val_str = os.getenv(name)
        if val_str is None:
            return int(default_ms)
        return max(0, int(val_str))
    except (ValueError, TypeError):
        return int(default_ms)


async def run_bounded(awaitable: Awaitable[Any], timeout_ms: int, *, on_timeout: Optional[Callable[[], None]] = None, cancel: bool = True) -> Any | None:
    """Await an awaitable with a time budget in milliseconds.

    - Returns the result on success.
    - Returns None on timeout (after optional cancellation and on_timeout callback).
    - Raises other exceptions from the awaitable as-is.
    
    This is a performance-critical function used throughout the runtime.
    Optimized for minimal overhead in the success path.
    """
    if timeout_ms is None or timeout_ms <= 0:
        return await awaitable
    
    timeout_s = max(0.001, float(timeout_ms) / 1000.0)
    
    # Always operate on a Task to avoid reusing a coroutine object
    task: asyncio.Task
    if isinstance(awaitable, asyncio.Task):
        task = awaitable
    else:
        loop = asyncio.get_running_loop()
        task = loop.create_task(awaitable)  # type: ignore[arg-type]
    
    # Wait without implicit cancellation to respect 'cancel' flag
    done, _ = await asyncio.wait({task}, timeout=timeout_s)
    
    if task in done:
        # Propagate result or exception
        return await task
    
    # Timeout occurred
    try:
        if cancel and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        if on_timeout:
            try:
                on_timeout()
            except Exception:
                pass
    
    return None


async def run_with_deadline(awaitable: Awaitable[T], timeout_ms: int, *, default: Optional[T] = None) -> BudgetResult:
    """Execute awaitable with deadline tracking and detailed result.
    
    Returns comprehensive result including timing and status.
    Useful for metrics and monitoring.
    """
    t0 = time.perf_counter()
    
    try:
        if timeout_ms is None or timeout_ms <= 0:
            value = await awaitable
            elapsed = (time.perf_counter() - t0) * 1000.0
            return BudgetResult(
                status=DeadlineStatus.SUCCESS,
                value=value,
                elapsed_ms=elapsed
            )
        
        timeout_s = max(0.001, float(timeout_ms) / 1000.0)
        
        # Create task
        task: asyncio.Task
        if isinstance(awaitable, asyncio.Task):
            task = awaitable
        else:
            task = asyncio.create_task(awaitable)  # type: ignore[arg-type]
        
        # Wait with timeout
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        elapsed = (time.perf_counter() - t0) * 1000.0
        
        if task in done:
            try:
                value = await task
                return BudgetResult(
                    status=DeadlineStatus.SUCCESS,
                    value=value,
                    elapsed_ms=elapsed
                )
            except asyncio.CancelledError:
                return BudgetResult(
                    status=DeadlineStatus.CANCELLED,
                    value=default,
                    elapsed_ms=elapsed
                )
            except Exception as e:
                return BudgetResult(
                    status=DeadlineStatus.ERROR,
                    value=default,
                    elapsed_ms=elapsed,
                    exception=e
                )
        else:
            # Timeout
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            
            return BudgetResult(
                status=DeadlineStatus.TIMEOUT,
                value=default,
                elapsed_ms=elapsed
            )
    
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return BudgetResult(
            status=DeadlineStatus.ERROR,
            value=default,
            elapsed_ms=elapsed,
            exception=e
        )
