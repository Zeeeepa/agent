from __future__ import annotations

import traceback
import re
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
        # Do not include the transcript in 'chains' since it is placed into <memory>
        chains, decay = build_chains("", err)
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
        # 3) <task> reflects the immediate objective: last user input or error to fix
        task_text = (err.strip() if err and err.strip() else (x.strip() if x and x.strip() else ""))

        # Assemble header parts if present
        header_parts = []
        if ctx:
            # ensure a trailing newline after the block
            header_parts.append(ctx.rstrip() + "\n")  # already wrapped as <embeddings_context> ...
        if mem_text and mem_text.strip():
            header_parts.append(f"<memory>\n\n{mem_text.strip()}\n\n</memory>\n")
        if task_text:
            header_parts.append(f"<task>\n\n{task_text}\n\n</task>\n")

        if header_parts:
            header_text = "\n\n".join(header_parts)
            # Normalize potential non-breaking spaces and enforce newlines between blocks
            # Replace any stray unicode spaces or characters between tag boundaries
            header_text = header_text.replace("\u00A0", " ")
            header_text = re.sub(r"(</embeddings_context>)[^<]*(<memory>)", r"\1\n\n\2", header_text)
            header_text = re.sub(r"(</memory>)[^<]*(<task>)", r"\1\n\n\2", header_text)
            chains = header_text + ("\n\n" + chains if chains else "")
        if decay:
            await dec_pulse(decay)
        # Final normalization guard: ensure blocks don't touch even if prior steps introduced spaces
        chains = chains.replace("\u00A0", " ")
        chains = re.sub(r"(</embeddings_context>)[^<]*(<memory>)", r"\1\n\n\2", chains)
        chains = re.sub(r"(</memory>)[^<]*(<task>)", r"\1\n\n\2", chains)
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
