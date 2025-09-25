"""Async runtime loop for the Jinx agent.

``pulse_core()`` wires input, spinner, and conversation services together and
drives the primary event loop. It is intentionally lightweight, cancellation
friendly, and side-effect contained for testability.
"""

from __future__ import annotations

import asyncio
import os
from jinx.banner_service import show_banner
from jinx.utils import chaos_patch
from jinx.runtime import start_input_task, frame_shift as _frame_shift
from jinx.embeddings import start_embeddings_task, start_project_embeddings_task
import jinx.state as jx_state
import contextlib
from jinx.memory.optimizer import stop as stop_memory_optimizer, start_memory_optimizer_task
from jinx.conversation.error_worker import stop_error_worker

async def pulse_core() -> None:
    """Run the main asynchronous processing loop.

    The loop:
    - Shows the startup banner.
    - Starts an input task that feeds a queue with user messages.
    - For each message, displays a spinner while executing the conversation step.
    - Gracefully cancels background tasks on shutdown or when the shutdown_event is set.
    """
    show_banner()
    # Bounded queue to avoid unbounded memory growth under bursts
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)

    async with chaos_patch():
        jobs: list[asyncio.Task[None]] = [
            start_input_task(q),
            asyncio.create_task(_frame_shift(q)),
            start_embeddings_task(),
            start_memory_optimizer_task(),
            # Always start project-wide embeddings generator by default
            start_project_embeddings_task(),
        ]
        shutdown_task: asyncio.Task | None = None
        try:
            # Race the running jobs against a shutdown signal
            shutdown_task = asyncio.create_task(jx_state.shutdown_event.wait())
            done, pending = await asyncio.wait({shutdown_task, *jobs}, return_when=asyncio.FIRST_COMPLETED)
            if shutdown_task in done:
                # Graceful shutdown requested (e.g., pulse <= 0)
                # Stop auxiliary background workers first
                with contextlib.suppress(Exception):
                    await stop_error_worker()
                    await stop_memory_optimizer()
                for x in jobs:
                    x.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
            else:
                # Jobs completed naturally; ensure we don't leak the waiter
                if shutdown_task is not None:
                    shutdown_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await shutdown_task
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Signal global shutdown to all components
            jx_state.shutdown_event.set()
            with contextlib.suppress(Exception):
                await stop_error_worker()
                await stop_memory_optimizer()
            for x in jobs:
                x.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        finally:
            # Ensure the shutdown waiter is not leaked
            st = shutdown_task
            if isinstance(st, asyncio.Task):
                if not st.done():
                    st.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await st
