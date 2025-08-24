"""Conversation service.

Coordinates model output parsing, execution, and error reporting.
"""

from __future__ import annotations

import shutil
import traceback
from typing import Optional
import textwrap
from jinx.logging_service import glitch_pulse, bomb_log
from jinx.openai_service import spark_openai
from jinx.error_service import dec_pulse
from jinx.conversation import build_chains, run_blocks
from jinx.sandbox.utils import read_latest_sandbox_tail


def pretty_echo(text: str, title: str = "Jinx") -> None:
    """Render model output in a neat ASCII box with a title.

    - Uses word-wrapping (no mid-word splits) for readability.
    - Preserves blank lines from the original text.
    - Avoids ANSI so it won't clash with prompt rendering.
    """
    width = shutil.get_terminal_size((80, 24)).columns
    width = max(50, min(width, 120))
    inner_w = width - 2

    # Title bar
    title_str = f" {title} " if title else ""
    title_len = len(title_str)
    # Ensure we have room for the title; fall back to plain bar if not
    if title_len and title_len + 2 < inner_w:
        top = "+-" + title_str + ("-" * (inner_w - title_len - 2)) + "+"
    else:
        top = "+" + ("-" * inner_w) + "+"
    bot = "+" + ("-" * inner_w) + "+"

    print(top)
    lines = text.splitlines() if text else [""]
    for ln in lines:
        # Wrap without breaking words or hyphenating; keep blank lines
        wrapped = (
            textwrap.wrap(
                ln,
                width=inner_w,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
            if ln.strip() != ""
            else [""]
        )
        for chunk in wrapped:
            pad = inner_w - len(chunk)
            print(f"|{chunk}{' ' * pad}|")
    print(bot + "\n")


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

async def show_sandbox_tail() -> None:
    """Print the latest sandbox log (full if short, else last N lines)."""
    content, _ = read_latest_sandbox_tail()
    if content is not None:
        pretty_echo(content, title="Jinx")


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
        async def on_exec_error(err_msg: Optional[str]) -> None:
            # Sandbox callback sends None on success — ignore to avoid duplicate log prints
            if not err_msg:
                return
            pretty_echo(out)
            await show_sandbox_tail()
            await corrupt_report(err_msg)

        executed = await run_blocks(out, code_id, on_exec_error)
        if not executed:
            await bomb_log("No executable <python_{key}> block found in model output; displaying raw output.")
            pretty_echo(out)
            await dec_pulse(10)
        else:
            # After successful execution, also surface the latest sandbox log context
            await show_sandbox_tail()
    except Exception:
        await bomb_log(traceback.format_exc())
        await dec_pulse(50)
