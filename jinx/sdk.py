from __future__ import annotations

"""
Jinx SDK — self-use API for autonomous orchestration and self-reprogramming.

This module exposes a minimal, stable API surface so internal programs (and
external Python callers, if needed) can interact with the running Jinx
instance without going through additional CLIs. It respects the constraint
that Jinx is started via `python jinx.py` only; this is just an importable
facade.
"""

import asyncio
from typing import Any, Optional

from jinx.micro.runtime import api as rt_api
from jinx.micro.runtime.patch.autopatch import AutoPatchArgs, autopatch
from jinx.sandbox.async_runner import run_sandbox


async def spawn(message: str, *, group: str = "main") -> str:
    """Submit a message into the runtime pipeline as a task.

    Returns a task id. This does not block on completion of the turn.
    """
    # Encode group tag inline to match frame_shift grouping semantics
    text = f"[#group:{group}] {message}" if group else message
    return await rt_api.submit_task("message.enqueue", text=text)


async def autopatch_apply(
    *,
    path: Optional[str] = None,
    code: Optional[str] = None,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    symbol: Optional[str] = None,
    anchor: Optional[str] = None,
    query: Optional[str] = None,
    preview: bool = False,
    max_span: Optional[int] = None,
    context_before: Optional[str] = None,
    context_tolerance: Optional[float] = None,
) -> tuple[bool, str, str]:
    """Apply an intelligent patch using the core autopatch orchestrator.

    This is a thin wrapper around `jinx.micro.runtime.patch.autopatch.autopatch`.
    """
    args = AutoPatchArgs(
        path=path,
        code=code,
        line_start=line_start,
        line_end=line_end,
        symbol=symbol,
        anchor=anchor,
        query=query,
        preview=preview,
        max_span=max_span,
        context_before=context_before,
        context_tolerance=context_tolerance,
    )
    return await autopatch(args)


async def sandbox_run(code: str) -> None:
    """Execute code in the isolated sandbox (process with hard timeout)."""
    await run_sandbox(code)


async def request_reprogram(goal: str, *, timeout_s: float = 30.0) -> None:
    """Ask the SelfReprogrammer to prepare and apply a blue-green self-update
    to achieve `goal`. Returns when the request has been enqueued; the
    blue-green switch will be handled by the self-update manager.
    """
    await rt_api.submit_task("reprogram.request", goal=goal, timeout_s=timeout_s)
