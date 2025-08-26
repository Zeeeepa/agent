from __future__ import annotations

"""Memory optimization pipeline.

Collects recent transcript and evergreen memory, asks the LLM to compact
and persist updated memory state, and serializes executions through a
single worker to preserve ordering.
"""

import os
import asyncio
from typing import Optional, Tuple
import re

from jinx.logging_service import bomb_log, glitch_pulse
from jinx.prompts import get_prompt
from jinx.openai_mod import call_openai
from jinx.retry import detonate_payload
from .parse import parse_output
from .storage import read_evergreen, write_state
from jinx.log_paths import OPENAI_REQUESTS_DIR_MEMORY
from jinx.logger.openai_requests import write_openai_request_dump
from jinx.config import ALL_TAGS
import jinx.state as jx_state
import contextlib

# Single worker ensures strict ordering; lock protects model call & writes
_mem_lock: asyncio.Lock = asyncio.Lock()
_queue: asyncio.Queue[Tuple[Optional[str], asyncio.Future[None]]] | None = None
_worker_task: asyncio.Task[None] | None = None
_stopping: bool = False


async def _optimize_memory_impl(snapshot: str | None) -> None:
    """Run a single memory optimization round.

    Parameters
    ----------
    snapshot : str | None
        Optional explicit transcript; when None, pulls from `glitch_pulse()`.
    """
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

        # Compose a structured input for the optimizer with clear tags and spacing
        parts: list[str] = []
        t_body = (transcript or "").strip()

        # Extract tool blocks (machine/python) out of transcript
        tool_blocks: list[str] = []
        if t_body:
            tag_alt = "|".join(sorted(ALL_TAGS))
            pattern = re.compile(fr"<(?:{tag_alt})_[^>]+>.*?</(?:{tag_alt})_[^>]+>", re.DOTALL)
            for m in pattern.finditer(t_body):
                tool_blocks.append(m.group(0).strip())
            # Remove tool blocks from transcript text
            t_body = pattern.sub("", t_body)
            # Normalize spacing inside transcript (collapse 3+ newlines to 2)
            t_body = re.sub(r"\n{3,}", "\n", t_body).strip()
            if t_body:
                parts.append(f"<transcript>\n{t_body}\n</transcript>")
        # Append evergreen immediately after transcript
        e_body = (evergreen or "").strip()
        if e_body:
            parts.append(f"<evergreen>\n{e_body}\n</evergreen>")

        # Then append extracted tool blocks, each separated clearly
        for blk in tool_blocks:
            # Ensure nice spacing around tool blocks
            cleaned = re.sub(r"\n{3,}", "\n", blk.strip())
            parts.append(cleaned)
        # Join with a full blank line between logical sections (<transcript>, <evergreen>, tool blocks)
        input_text = ("\n\n".join(parts)).replace("\u00A0", " ")

        async def _invoke_llm() -> str:
            # Log memory-optimizer request via micro-module
            try:
                await write_openai_request_dump(
                    target_dir=OPENAI_REQUESTS_DIR_MEMORY,
                    kind="MEMORY",
                    instructions=instructions,
                    input_text=input_text,
                    model=model,
                )
            except Exception:
                pass
            return await call_openai(instructions, model, input_text)

        # Reuse shared retry/timeout helper for consistency with openai_service
        out = await detonate_payload(_invoke_llm, timeout=timeout_sec)

        compact, durable = parse_output(out)
        await write_state(compact, durable)
        await bomb_log("MEMORY optimize: done")
    except Exception as e:
        await bomb_log(f"ERROR memory optimize failed: {e}")


async def _worker_loop() -> None:
    assert _queue is not None
    get_task: asyncio.Task | None = None
    shutdown_task: asyncio.Task | None = None
    try:
        while True:
            # Fast-exit if shutdown already requested
            if jx_state.shutdown_event.is_set():
                break
            # Wait for either a queue item or a shutdown signal
            get_task = asyncio.create_task(_queue.get())
            shutdown_task = asyncio.create_task(jx_state.shutdown_event.wait())
            done, pending = await asyncio.wait({get_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED)
            # If shutdown requested, exit loop gracefully
            if shutdown_task in done:
                if get_task is not None:
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
                break
            # Otherwise process the queued job
            if shutdown_task is not None:
                shutdown_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await shutdown_task
            snapshot, fut = get_task.result()
            try:
                # Serialize through the memory lock
                async with _mem_lock:
                    await _optimize_memory_impl(snapshot)
                with contextlib.suppress(BaseException):
                    if not fut.done():
                        fut.set_result(None)
            except Exception as e:  # propagate to caller
                with contextlib.suppress(BaseException):
                    if not fut.done():
                        fut.set_exception(e)
            finally:
                with contextlib.suppress(Exception):
                    _queue.task_done()
    except asyncio.CancelledError:
        # Exit quietly on cancellation
        pass
    finally:
        # Ensure any locally created tasks are cancelled/awaited to avoid leaks
        if get_task is not None and not get_task.done():
            get_task.cancel()
            # During interpreter teardown, awaiting may raise RuntimeError: loop closed / no running loop
            with contextlib.suppress(asyncio.CancelledError, RuntimeError, BaseException):
                await get_task
        if shutdown_task is not None and not shutdown_task.done():
            shutdown_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError, BaseException):
                await shutdown_task


def _ensure_worker() -> None:
    global _queue, _worker_task
    if _queue is None:
        _queue = asyncio.Queue(maxsize=32)
    # Do not start during shutdown or when stopping
    if jx_state.shutdown_event.is_set() or _stopping:
        return
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop(), name="memory-optimizer")


async def submit(snapshot: str | None = None) -> None:
    """Submit a memory optimization job and await its completion.

    Maintains strict FIFO ordering while running in a dedicated worker task.
    """
    # Do not enqueue new work if shutdown has been requested
    if jx_state.shutdown_event.is_set():
        return
    _ensure_worker()
    assert _queue is not None
    fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    await _queue.put((snapshot, fut))
    await fut


async def stop() -> None:
    """Stop the optimizer worker and cancel pending jobs gracefully."""
    global _worker_task, _stopping
    _stopping = True
    # Signal is already set by shutdown path; ensure worker exits and drain queue
    if _queue is not None:
        # Drain any queued items and cancel their futures so callers don't hang
        try:
            while True:
                snapshot, fut = _queue.get_nowait()
                if not fut.done():
                    fut.set_exception(asyncio.CancelledError())
                _queue.task_done()
        except asyncio.QueueEmpty:
            pass
    if _worker_task is not None and not _worker_task.done():
        _worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, RuntimeError, BaseException):
            await _worker_task
    _worker_task = None
    _stopping = False


def start_memory_optimizer_task() -> asyncio.Task[None]:
    """Start the memory optimizer background worker and return its task.

    This mirrors the micro-module pattern used by embeddings service.
    Safe to call multiple times; will return the existing task if already running.
    """
    _ensure_worker()
    if _worker_task is None:
        # During shutdown, return a completed noop task to satisfy the contract
        return asyncio.create_task(asyncio.sleep(0), name="memory-optimizer-noop")
    return _worker_task
