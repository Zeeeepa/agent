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

# Single worker ensures strict ordering; lock protects model call & writes
_mem_lock: asyncio.Lock = asyncio.Lock()
_queue: asyncio.Queue[Tuple[Optional[str], asyncio.Future[None]]] | None = None
_worker_task: asyncio.Task[None] | None = None


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
            t_body = re.sub(r"\n{3,}", "\n\n", t_body).strip()
            if t_body:
                parts.append(f"<transcript>\n\n{t_body}\n\n</transcript>")
        # Append evergreen immediately after transcript
        e_body = (evergreen or "").strip()
        if e_body:
            parts.append(f"<evergreen>\n\n{e_body}\n\n</evergreen>")

        # Then append extracted tool blocks, each separated clearly
        for blk in tool_blocks:
            # Ensure nice spacing around tool blocks
            cleaned = re.sub(r"\n{3,}", "\n\n", blk.strip())
            parts.append(cleaned)
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
    fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    await _queue.put((snapshot, fut))
    await fut
