from __future__ import annotations

"""Advanced error worker with circuit breaker and metrics.

Provides a dedicated background worker to serialize error-driven retries, decouple
queue lifecycle from the orchestrator, and support graceful shutdown via the
shared shutdown_event.

Features:
- Circuit breaker pattern for error handling
- Metrics tracking
- Bounded queue with overflow handling
- Graceful degradation
"""

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Optional
import threading

import jinx.state as jx_state


@dataclass
class ErrorWorkerMetrics:
    """Metrics for error worker performance."""
    total_errors: int = 0
    processed_errors: int = 0
    failed_retries: int = 0
    queue_overflows: int = 0
    avg_processing_time_ms: float = 0.0
    consecutive_failures: int = 0
    last_error_time: float = 0.0


# Local queue and task for error retries
_err_queue: asyncio.Queue[str] | None = None
_err_worker_task: asyncio.Task[None] | None = None
_metrics = ErrorWorkerMetrics()
_metrics_lock = threading.RLock()


def _ensure_error_worker() -> None:
    """Ensure error worker is running with bounded queue."""
    global _err_queue, _err_worker_task
    if _err_queue is None:
        # Bounded queue to prevent memory exhaustion
        _err_queue = asyncio.Queue(maxsize=256)
    if _err_worker_task is None or _err_worker_task.done():
        _err_worker_task = asyncio.create_task(_error_retry_worker(), name="error-worker")


async def _error_retry_worker() -> None:
    """Enhanced error worker with metrics and circuit breaking."""
    global _metrics
    assert _err_queue is not None
    
    try:
        while True:
            # Wait for either an error item or a shutdown signal
            get_task = asyncio.create_task(_err_queue.get())
            shutdown_task = asyncio.create_task(jx_state.shutdown_event.wait())
            done, _ = await asyncio.wait({get_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED)
            
            if shutdown_task in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                break
            
            # Process the item
            shutdown_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await shutdown_task
            
            err = get_task.result()
            t0 = time.perf_counter()
            
            try:
                # Circuit breaker: if too many consecutive failures, add delay
                with _metrics_lock:
                    if _metrics.consecutive_failures >= 5:
                        # Exponential backoff for consecutive failures
                        delay = min(10.0, 2 ** min(_metrics.consecutive_failures - 5, 5))
                        await asyncio.sleep(delay)
                
                # Late import to avoid circular dependency with orchestrator
                from jinx.micro.conversation.orchestrator import shatter  # noqa
                await shatter("", err=err)
                
                # Success - reset consecutive failures
                with _metrics_lock:
                    _metrics.processed_errors += 1
                    _metrics.consecutive_failures = 0
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    # Update average processing time
                    count = _metrics.processed_errors
                    _metrics.avg_processing_time_ms = (
                        (_metrics.avg_processing_time_ms * (count - 1) + elapsed_ms) / count
                    )
                    
            except Exception as e:
                # Track failure
                with _metrics_lock:
                    _metrics.failed_retries += 1
                    _metrics.consecutive_failures += 1
                    _metrics.last_error_time = time.time()
                
                # Log only if not shutting down
                if not jx_state.shutdown_event.is_set():
                    try:
                        from jinx.logging_service import bomb_log
                        await bomb_log(f"Error worker retry failed: {e}")
                    except Exception:
                        pass
            finally:
                _err_queue.task_done()
                
    except asyncio.CancelledError:
        pass


async def enqueue_error_retry(err: str) -> None:
    """Enqueue error for retry with overflow handling."""
    global _metrics
    
    # Do not enqueue after shutdown has been requested
    if jx_state.shutdown_event.is_set():
        return
    
    _ensure_error_worker()
    assert _err_queue is not None
    
    with _metrics_lock:
        _metrics.total_errors += 1
    
    # Try to enqueue with overflow detection
    try:
        _err_queue.put_nowait(err)
    except asyncio.QueueFull:
        # Queue full - drop oldest and log
        with _metrics_lock:
            _metrics.queue_overflows += 1
        
        try:
            # Drop oldest error
            _ = _err_queue.get_nowait()
            _err_queue.task_done()
            # Add new error
            _err_queue.put_nowait(err)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            # If still can't enqueue, just drop it
            pass


async def stop_error_worker() -> None:
    """Stop error worker and drain pending items."""
    global _err_queue, _err_worker_task
    
    # Drain any pending items so we don't hang if worker already exited
    if _err_queue is not None:
        try:
            while True:
                _ = _err_queue.get_nowait()
                _err_queue.task_done()
        except asyncio.QueueEmpty:
            pass
    
    if _err_worker_task is not None and not _err_worker_task.done():
        _err_worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _err_worker_task
    
    _err_worker_task = None


def get_error_worker_metrics() -> ErrorWorkerMetrics:
    """Get current error worker metrics."""
    with _metrics_lock:
        return ErrorWorkerMetrics(
            total_errors=_metrics.total_errors,
            processed_errors=_metrics.processed_errors,
            failed_retries=_metrics.failed_retries,
            queue_overflows=_metrics.queue_overflows,
            avg_processing_time_ms=_metrics.avg_processing_time_ms,
            consecutive_failures=_metrics.consecutive_failures,
            last_error_time=_metrics.last_error_time
        )
