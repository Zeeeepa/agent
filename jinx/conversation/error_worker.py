from __future__ import annotations

"""Error worker micro-module.

Provides a dedicated background worker to serialize error-driven retries, decouple
queue lifecycle from the orchestrator, and support graceful shutdown via the
shared shutdown_event.
"""

import asyncio
import contextlib

import jinx.state as jx_state

# Local queue and task for error retries
_err_queue: asyncio.Queue[str] | None = None
_err_worker_task: asyncio.Task[None] | None = None


def _ensure_error_worker() -> None:
    global _err_queue, _err_worker_task
    if _err_queue is None:
        _err_queue = asyncio.Queue(maxsize=256)
    if _err_worker_task is None or _err_worker_task.done():
        _err_worker_task = asyncio.create_task(_error_retry_worker())


async def _error_retry_worker() -> None:
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
            try:
                # Late import to avoid circular dependency
                from .orchestrator import shatter  # noqa
                await shatter("", err=err)
            except Exception:
                # Swallow to keep worker alive; actual error path logs separately
                pass
            finally:
                _err_queue.task_done()
    except asyncio.CancelledError:
        pass


async def enqueue_error_retry(err: str) -> None:
    # Do not enqueue after shutdown has been requested
    if jx_state.shutdown_event.is_set():
        return
    _ensure_error_worker()
    assert _err_queue is not None
    await _err_queue.put(err)


async def stop_error_worker() -> None:
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
