"""Interactive input service.

Provides an async prompt using prompt_toolkit and pushes sanitized user input
into an asyncio queue. Includes an inactivity watchdog that emits
"<no_response>" after a configurable timeout to keep the agent responsive.
"""

from __future__ import annotations

import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from .state import boom_limit
from .logging_service import blast_mem, bomb_log
from jinx.log_paths import TRIGGER_ECHOES, BLUE_WHISPERS
from jinx.async_utils.queue import try_put_nowait, put_drop_oldest


async def neon_input(qe: asyncio.Queue[str]) -> None:
    """Read user input and feed it into the provided queue.

    Parameters
    ----------
    qe : asyncio.Queue[str]
        Target queue for sanitized user input.
    """
    finger_wire = KeyBindings()
    sess = PromptSession(key_bindings=finger_wire)
    boom_clock: dict[str, float] = {"time": asyncio.get_running_loop().time()}
    activity = asyncio.Event()

    @finger_wire.add("<any>")
    def _(triggerbit) -> None:  # prompt_toolkit callback
        boom_clock["time"] = asyncio.get_running_loop().time()
        triggerbit.app.current_buffer.insert_text(triggerbit.key_sequence[0].key)
        # Signal activity to reset the inactivity timer immediately
        activity.set()

    async def kaboom_watch() -> None:
        """Emit <no_response> after inactivity using a reactive timer.

        Avoids periodic polling by waiting for either activity or timeout.
        """
        while True:
            # Calculate remaining time based on last activity
            now = asyncio.get_running_loop().time()
            remaining = max(0.0, boom_limit - (now - boom_clock["time"]))
            activity.clear()
            try:
                # Wait for either new activity or the inactivity timeout
                await asyncio.wait_for(activity.wait(), timeout=remaining)
                # Activity occurred: loop to recalculate remaining
                continue
            except asyncio.TimeoutError:
                # Timeout: no activity within boom_limit
                await blast_mem("<no_response>")
                await bomb_log("<no_response>", TRIGGER_ECHOES)
                # Do not disrupt FIFO order: emit only if queue has space
                placed = try_put_nowait(qe, "<no_response>")
                if not placed:
                    await bomb_log("<no_response> skipped: input queue saturated", BLUE_WHISPERS)
                boom_clock["time"] = asyncio.get_running_loop().time()

    watch_task = asyncio.create_task(kaboom_watch())
    try:
        while True:
            try:
                # key_bindings already provided to PromptSession
                v: str = await sess.prompt_async("\n")
                if v.strip():
                    # Do not write to transcript here; conversation_service will append
                    await bomb_log(v, TRIGGER_ECHOES)
                    # Preserve strict FIFO ordering for user inputs via backpressure
                    await qe.put(v.strip())
            except EOFError:
                break
            except Exception as e:  # pragma: no cover - guard rail for TTY issues
                await bomb_log(f"ERROR INPUT chaos keys went rogue: {e}")
    finally:
        # Ensure watchdog is cancelled when input loop exits
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
