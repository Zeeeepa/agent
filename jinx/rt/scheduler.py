from __future__ import annotations

"""
Deadline-aware scheduling helpers for Jinx turns.

This is a lightweight, cooperative layer that timeboxes tasks according to
supplied deadlines. It does not try to preempt running Python coroutines; it
wraps them with asyncio.wait_for to enforce deadlines.

Future: upgrade to a proper EDF queue with admission control.
"""

import asyncio
from typing import Awaitable, Callable, Optional


def schedule_turn(factory: Callable[[], Awaitable[None]], *, deadline_ms: int, name: Optional[str] = None) -> asyncio.Task[None]:
    """Schedule a turn coroutine with a hard timeout (deadline in ms).

    Returns the created asyncio.Task so callers can track completion.
    """
    timeout_s = max(0.05, deadline_ms / 1000.0)

    async def _runner() -> None:
        try:
            await asyncio.wait_for(factory(), timeout=timeout_s)
        except asyncio.TimeoutError:
            # Expose a friendly cancellation for upstream metrics; suppress traceback
            return None
        except asyncio.CancelledError:
            # Treat cancellations like timeouts in RT pipeline
            return None

    return asyncio.create_task(_runner(), name=name or "turn")
