from __future__ import annotations

import traceback
import re
import asyncio
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
from jinx.conversation.formatting import build_header, ensure_header_block_separation
from jinx.memory import read_evergreen
from jinx.config import ALL_TAGS
from jinx.conversation.error_worker import enqueue_error_retry
import jinx.state as jx_state


async def corrupt_report(err: Optional[str]) -> None:
    """Log an error, enqueue a serialized retry, and decay pulse."""
    if err is None:
        return
    await bomb_log(err)
    # Enqueue follow-up step to be processed by a single worker
    await enqueue_error_retry(err)
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
        # Do not include the transcript in 'chains' since it is placed into <memory>
        # Do not inject error text into the body chains; it will live in <error>
        chains, decay = build_chains("", None)
        # Build standardized header blocks in a stable order before the main chains
        # 1) <embeddings_context> from recent dialogue/sandbox using current input as query
        try:
            ctx = await build_context_for(x or synth or "")
        except Exception:
            ctx = ""
        # 2) <memory> from transcript (exclude the latest user input line if present)
        mem_text = synth or ""
        try:
            last_line = (x or "").strip()
            if last_line and mem_text.strip().endswith(last_line):
                lines = [ln for ln in mem_text.splitlines()]
                # remove only the final occurrence if it matches exactly
                if lines and lines[-1].strip() == last_line:
                    lines = lines[:-1]
                mem_text = "\n".join(lines).strip()
        except Exception:
            pass
        # Sanitize transcript for <memory>: remove tool blocks and prior header blocks
        try:
            tag_alt = "|".join(sorted(ALL_TAGS))
            # Remove tool blocks like <machine_...>...</machine_...>, <python_...>...</python_...>
            tool_pat = re.compile(fr"<(?:{tag_alt})_[^>]+>.*?</(?:{tag_alt})_[^>]+>", re.DOTALL)
            mem_text = tool_pat.sub("", mem_text)
            # Remove any prior header blocks to avoid nesting/duplication
            header_pat = re.compile(r"<(?:embeddings_context|memory|evergreen|task|error)>.*?</(?:embeddings_context|memory|evergreen|task|error)>", re.DOTALL)
            mem_text = header_pat.sub("", mem_text)
            # Collapse excessive whitespace/newlines
            mem_text = re.sub(r"\n{3,}", "\n\n", mem_text).strip()
        except Exception:
            pass
        # 2.5) <evergreen> persistent durable facts
        try:
            evergreen_text = (await read_evergreen()) or ""
        except Exception:
            evergreen_text = ""
        # 3) <task> reflects the immediate objective: when handling an error,
        #    avoid copying traceback or transcript into <task>
        task_text = ("" if (err and err.strip()) else ((x.strip() if x and x.strip() else "")))
        # Optional <error> block carries execution or prior error details
        error_text = (err.strip() if err and err.strip() else None)

        # Assemble header using shared formatting utilities
        header_text = build_header(ctx, mem_text, task_text, error_text, evergreen_text)
        if header_text:
            chains = header_text + ("\n\n" + chains if chains else "")
        # If an error is present, enforce a decay hit to drive auto-fix loop
        if err and err.strip():
            decay = max(decay, 50)
        if decay:
            await dec_pulse(decay)
        # Final normalization guard
        chains = ensure_header_block_separation(chains)
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
