"""Conversation service.

Coordinates model output parsing, execution, and error reporting.
"""

from __future__ import annotations

import traceback
from typing import Optional
from jinx.logging_service import glitch_pulse, bomb_log
from jinx.openai_service import spark_openai
from jinx.error_service import dec_pulse
from jinx.conversation import build_chains, run_blocks


async def corrupt_report(err: Optional[str]) -> None:
    """Log an error, echo it into the conversation, and decay pulse.

    Parameters
    ----------
    err : Optional[str]
        Error text to report. If None, function returns immediately.
    """
    if err is None:
        return
    await bomb_log(err)
    trail = await glitch_pulse()
    if trail:
        await shatter(trail + f"\n{err}", err=err)
    await dec_pulse(30)


async def shatter(x: str, err: Optional[str] = None) -> None:
    """Drive a single conversation step and optionally handle an error context.

    Parameters
    ----------
    x : str
        Not used directly; conversation state is read from the transcript.
        Kept for compatibility with the orchestrator signature.
    err : Optional[str]
        Optional recent error message to include for context.
    """
    try:
        synth = await glitch_pulse()
        chains, decay = build_chains(synth, err)
        if decay:
            await dec_pulse(decay)
        out, code_id = await spark_openai(chains)

        # Ensure that on any execution error we also show the raw model output
        async def on_exec_error(err_msg: str) -> None:
            print(out)
            await corrupt_report(err_msg)

        executed = await run_blocks(out, code_id, on_exec_error)
        if not executed:
            await bomb_log("No executable <python_{key}> block found in model output; displaying raw output.")
            print(out)
            await dec_pulse(10)
    except Exception:
        await bomb_log(traceback.format_exc())
        await dec_pulse(50)
