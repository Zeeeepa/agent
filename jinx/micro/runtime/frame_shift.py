from __future__ import annotations

import asyncio
from jinx.conversation.orchestrator import shatter
from jinx.spinner_service import sigil_spin


async def frame_shift(q: asyncio.Queue[str]) -> None:
    """Process queue items, wrapping each conversation step with a spinner."""
    evt: asyncio.Event = asyncio.Event()
    while True:
        c: str = await q.get()
        evt.clear()
        spintask = asyncio.create_task(sigil_spin(evt))
        try:
            await shatter(c)
        finally:
            evt.set()
            await spintask
