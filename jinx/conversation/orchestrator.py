from __future__ import annotations

import traceback
from typing import Optional

from jinx.logging_service import glitch_pulse, bomb_log, blast_mem
from jinx.openai_service import spark_openai
from jinx.error_service import dec_pulse
from jinx.conversation import build_chains, run_blocks
from jinx.sandbox.utils import read_latest_sandbox_tail
from jinx.memory_service import optimize_memory
from .ui import pretty_echo
from jinx.embeddings.retrieval import build_context_for
from jinx.embeddings.pipeline import embed_text


async def corrupt_report(err: Optional[str]) -> None:
    """Log an error, echo it into the conversation, and decay pulse."""
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
    """Drive a single conversation step and optionally handle an error context."""
    try:
        # Append the user input to the transcript first to ensure ordering
        if x and x.strip():
            await blast_mem(x.strip())
            # Also embed the raw user input for retrieval (source: dialogue)
            try:
                await embed_text(x.strip(), source="dialogue", kind="user")
            except Exception:
                pass
        synth = await glitch_pulse()
        chains, decay = build_chains(synth, err)
        # Prepend embeddings-based context using the current user input as the query
        try:
            ctx = await build_context_for(x or synth or "")
            if ctx:
                # ctx already formatted as <context_from_embeddings>...</context_from_embeddings>
                chains = f"{ctx}\n\n{chains}"
        except Exception:
            # Fail open if retrieval has issues
            pass
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
        # Append the model output to the transcript to keep turn-complete context
        if out and out.strip():
            await blast_mem(out.strip())
            # Also embed the agent output for retrieval (source: dialogue)
            try:
                await embed_text(out.strip(), source="dialogue", kind="agent")
            except Exception:
                pass
    except Exception:
        await bomb_log(traceback.format_exc())
        await dec_pulse(50)
    finally:
        # Run memory optimization after each model interaction using a per-turn snapshot
        snap = await glitch_pulse()
        await optimize_memory(snap)
