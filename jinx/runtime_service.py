"""Async runtime loop for the Jinx agent.

``pulse_core()`` wires input, spinner, and conversation services together and
drives the primary event loop. It is intentionally lightweight, cancellation
friendly, and side-effect contained for testability.
"""

from __future__ import annotations

import asyncio
from jinx.banner_service import show_banner
from jinx.utils import chaos_patch
from jinx.runtime import start_input_task, frame_shift as _frame_shift
from jinx.embeddings import start_embeddings_task


async def pulse_core() -> None:
    """Run the main asynchronous processing loop.

    The loop:
    - Shows the startup banner.
    - Starts an input task that feeds a queue with user messages.
    - For each message, displays a spinner while executing the conversation step.
    - Gracefully cancels background tasks on shutdown.
    """
    show_banner()
    # Bounded queue to avoid unbounded memory growth under bursts
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)

    async with chaos_patch():
        jobs: list[asyncio.Task[None]] = [
            start_input_task(q),
            asyncio.create_task(_frame_shift(q)),
            start_embeddings_task(),
        ]
        try:
            await asyncio.gather(*jobs)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for x in jobs:
                x.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
