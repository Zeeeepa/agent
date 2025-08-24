from __future__ import annotations

import os
import asyncio
from typing import Optional, Tuple

from jinx.logging_service import bomb_log, glitch_pulse
from jinx.prompts import get_prompt
from jinx.openai_mod import call_openai
from .parse import parse_output
from .storage import read_evergreen, write_state

# Single worker ensures strict ordering; lock protects model call & writes
_mem_lock: asyncio.Lock = asyncio.Lock()
_queue: asyncio.Queue[Tuple[Optional[str], asyncio.Future[None]]] | None = None
_worker_task: asyncio.Task[None] | None = None


async def _optimize_memory_impl(snapshot: str | None) -> None:
    await bomb_log("MEMORY optimize: start")
    try:
        transcript = await glitch_pulse() if snapshot is None else snapshot
        evergreen = await read_evergreen()

        if not transcript and not evergreen:
            await bomb_log("MEMORY optimize: skip (empty state)")
            return

        instructions = get_prompt("memory_optimizer")
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        timeout_sec = float(os.getenv("MEMORY_TIMEOUT_SEC", "60"))

        input_text = (transcript or "") + ("\n\n" if transcript and evergreen else "") + (evergreen or "")
        try:
            # Apply a timeout to avoid stalling the whole pipeline
            async with asyncio.timeout(timeout_sec):
                out = await call_openai(instructions, model, input_text)
        except TimeoutError:
            await bomb_log(f"ERROR memory optimize timeout after {timeout_sec}s")
            return
        compact, durable = parse_output(out)
        await write_state(compact, durable)
        await bomb_log("MEMORY optimize: done")
    except Exception as e:
        await bomb_log(f"ERROR memory optimize failed: {e}")


async def _worker_loop() -> None:
    assert _queue is not None
    while True:
        snapshot, fut = await _queue.get()
        try:
            # Serialize through the memory lock
            async with _mem_lock:
                await _optimize_memory_impl(snapshot)
            if not fut.done():
                fut.set_result(None)
        except Exception as e:  # propagate to caller
            if not fut.done():
                fut.set_exception(e)
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _queue, _worker_task
    if _queue is None:
        _queue = asyncio.Queue(maxsize=32)
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())


async def submit(snapshot: str | None = None) -> None:
    """Submit a memory optimization job and await its completion.

    Maintains strict FIFO ordering while running in a dedicated worker task.
    """
    _ensure_worker()
    assert _queue is not None
    fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    await _queue.put((snapshot, fut))
    await fut
