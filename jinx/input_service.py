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
    boom_clock: dict[str, float] = {"time": asyncio.get_event_loop().time()}

    @finger_wire.add("<any>")
    def _(triggerbit) -> None:  # prompt_toolkit callback
        boom_clock["time"] = asyncio.get_event_loop().time()
        triggerbit.app.current_buffer.insert_text(triggerbit.key_sequence[0].key)

    async def kaboom_watch() -> None:
        while True:
            await asyncio.sleep(1)
            tick_tock = asyncio.get_event_loop().time()
            if tick_tock - boom_clock["time"] > boom_limit:
                await blast_mem("<no_response>")
                await bomb_log("<no_response>", TRIGGER_ECHOES)
                placed = try_put_nowait(qe, "<no_response>")
                if not placed:
                    await bomb_log("<no_response> dropped: input queue saturated", BLUE_WHISPERS)
                boom_clock["time"] = tick_tock

    asyncio.create_task(kaboom_watch())
    while True:
        try:
            v: str = await sess.prompt_async("\n", key_bindings=finger_wire)
            if v.strip():
                await blast_mem(v)
                await bomb_log(v, TRIGGER_ECHOES)
                put_drop_oldest(
                    qe,
                    v.strip(),
                    on_drop=lambda: bomb_log("input dropped oldest: queue saturated", BLUE_WHISPERS),
                )
        except EOFError:
            break
        except Exception as e:  # pragma: no cover - guard rail for TTY issues
            await bomb_log(f"ERROR INPUT chaos keys went rogue: {e}")
