from __future__ import annotations

import asyncio
from jinx.input_service import neon_input


def start_input_task(q: asyncio.Queue[str]) -> asyncio.Task[None]:
    """Start the input task that feeds user messages into the queue."""
    return asyncio.create_task(neon_input(q))
