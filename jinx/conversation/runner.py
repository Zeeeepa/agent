from __future__ import annotations

from typing import Awaitable, Callable

from jinx.parser_service import parse_tagged_blocks, is_code_tag
from jinx.logging_service import bomb_log, blast_mem
from jinx.exec_service import spike_exec
from jinx.safety import chaos_taboo
from jinx.error_service import inc_pulse


async def run_blocks(raw_output: str, code_id: str, on_exec_error: Callable[[str], Awaitable[None]]) -> bool:
    """Parse model output, run the first executable code block, and bump pulse.

    Returns True if an executable block was found and executed, else False.
    """
    await bomb_log(f"\n{raw_output}\n")
    match = parse_tagged_blocks(raw_output, code_id)
    for tag, core in match:
        if is_code_tag(tag):
            await blast_mem(core)
            await spike_exec(core, chaos_taboo, on_exec_error)
            await inc_pulse(10)
            return True
    return False
