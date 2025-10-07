from __future__ import annotations

import asyncio
from jinx.conversation.orchestrator import shatter
from jinx.spinner_service import sigil_spin
import jinx.state as jx_state


async def frame_shift(q: asyncio.Queue[str]) -> None:
    """Process queue items, wrapping each conversation step with a spinner."""
    evt: asyncio.Event = asyncio.Event()
    while True:
        # Respect global shutdown
        if jx_state.shutdown_event.is_set():
            return
        # Soft-throttle: pause intake while the system is saturated
        while jx_state.throttle_event.is_set():
            if jx_state.shutdown_event.is_set():
                return
            await asyncio.sleep(0.05)
        c: str = await q.get()
        evt.clear()
        spintask = asyncio.create_task(sigil_spin(evt))
        try:
            await shatter(c)
        finally:
            evt.set()
            await spintask
