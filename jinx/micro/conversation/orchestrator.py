from __future__ import annotations

import traceback
from typing import Optional
import os

from jinx.logging_service import glitch_pulse, bomb_log, blast_mem
from jinx.openai_service import spark_openai
from jinx.error_service import dec_pulse
from jinx.conversation import build_chains, run_blocks
from jinx.micro.ui.output import pretty_echo
from jinx.micro.conversation.sandbox_view import show_sandbox_tail
from jinx.micro.conversation.error_report import corrupt_report
from jinx.embeddings.retrieval import build_context_for
from jinx.embeddings.project_retrieval import build_project_context_for
from jinx.embeddings.pipeline import embed_text
from jinx.conversation.formatting import build_header, ensure_header_block_separation
from jinx.micro.memory.storage import read_evergreen
from jinx.micro.conversation.memory_sanitize import sanitize_transcript_for_memory
from jinx.micro.embeddings.project_config import ENABLE as PROJ_EMB_ENABLE
from jinx.micro.embeddings.project_paths import PROJECT_FILES_DIR


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
        # 1) <embeddings_context> from recent dialogue/sandbox using current input as query,
        #    plus project code embeddings context assembled from emb/ when available
        try:
            base_ctx = await build_context_for(x or synth or "")
        except Exception:
            base_ctx = ""
        # Only build project context when enabled and emb/files exists to avoid unnecessary API calls
        proj_ctx = ""
        if PROJ_EMB_ENABLE and os.path.isdir(PROJECT_FILES_DIR):
            try:
                proj_ctx = await build_project_context_for(x or synth or "")
            except Exception:
                proj_ctx = ""
        ctx = "\n".join([c for c in [base_ctx, proj_ctx] if c])
        # 2) <memory> from transcript (exclude the latest user input line and sanitize)
        mem_text = sanitize_transcript_for_memory(synth or "", (x or "").strip())
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
        # Use a dedicated recovery prompt only when fixing an error; otherwise default prompt
        prompt_override = "burning_logic_recovery" if (err and err.strip()) else None
        out, code_id = await spark_openai(chains, prompt_override=prompt_override)

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
            await bomb_log(f"No executable <python_{code_id}> block found in model output; displaying raw output.")
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
        # Late import avoids circular import during startup
        from jinx.micro.memory.optimizer import submit as _opt_submit
        await _opt_submit(snap)
