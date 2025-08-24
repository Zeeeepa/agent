from __future__ import annotations

import asyncio
from jinx.conversation_service import shatter
from jinx.runtime.spinner_task import start_spinner


async def frame_shift(q: asyncio.Queue[str]) -> None:
    """Process queue items, wrapping each conversation step with a spinner."""
    evt: asyncio.Event = asyncio.Event()
    while True:
        c: str = await q.get()
        evt.clear()
        spintask = start_spinner(evt)
        try:
            await shatter(c)
        finally:
            evt.set()
            await spintask
