from __future__ import annotations

import asyncio
import os
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
        # Env gate to disable spinner if needed for diagnostics or TTY performance
        try:
            _spin_on = str(os.getenv("JINX_SPINNER_ENABLE", "1")).strip().lower() not in ("", "0", "false", "off", "no")
        except Exception:
            _spin_on = True
        spintask = asyncio.create_task(sigil_spin(evt)) if _spin_on else asyncio.create_task(asyncio.sleep(0))
        try:
            await shatter(c)
        finally:
            evt.set()
            await spintask
