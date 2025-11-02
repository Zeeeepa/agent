from __future__ import annotations

import os
from jinx.openai_mod import build_header_and_tag
from .openai_caller import call_openai, call_openai_validated, call_openai_stream_first_block
from jinx.log_paths import OPENAI_REQUESTS_DIR_GENERAL
from jinx.logger.openai_requests import write_openai_request_dump, write_openai_response_append
from jinx.micro.memory.storage import write_token_hint
from jinx.retry import detonate_payload
from .prompt_compose import compose_dynamic_prompt
from .macro_registry import MacroContext, expand_dynamic_macros
from .macro_providers import register_builtin_macros
from .macro_plugins import load_macro_plugins
from jinx.micro.conversation.cont import load_last_anchors
from jinx.micro.runtime.api import list_programs
import platform
import sys
import datetime as _dt
from .prompt_filters import sanitize_prompt_for_external_api
from jinx.micro.text.heuristics import is_code_like as _is_code_like
import asyncio as _asyncio
from jinx.micro.rt.timing import timing_section
from jinx.micro.embeddings.unified_context import build_unified_context_for
from jinx.micro.embeddings.context_compact import compact_context
from jinx.micro.llm.enrichers import auto_context_lines
from jinx.micro.llm.enrichers.exports import (
    patch_exports_lines as _patch_exports_lines,
    verify_exports_lines as _verify_exports_lines,
    run_exports_lines as _run_exports_lines,
)


async def code_primer(prompt_override: str | None = None) -> tuple[str, str]:
    """Build instruction header and return it with a code tag identifier.

    Returns (header_plus_prompt, code_tag_id).
    """
    return await build_header_and_tag(prompt_override)


async def _prepare_request(txt: str, *, prompt_override: str | None = None) -> tuple[str, str, str, str, str]:
    """Compose instructions and return (jx, tag, model, sx, stxt)."""
    jx, tag = await code_primer(prompt_override)
    # Cooperative yield helpers (env-gated)
    def _yield_on() -> bool:
        try:
            return str(os.getenv("JINX_COOP_YIELD", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            return True
    async def _yield0() -> None:
        if _yield_on():
            try:
                await _asyncio.sleep(0)
            except Exception:
                pass
    # Expand dynamic prompt macros in real time (vars/env/anchors/sys/runtime/exports + custom providers)
    try:
        jx = await compose_dynamic_prompt(jx, key=tag)
        await _yield0()
        # Unified embeddings context (code+brain+refs+graph+memory)
        try:
            _ctx = await build_unified_context_for(txt or "", max_chars=None, max_time_ms=300)
        except Exception:
            _ctx = ""
        have_unified_ctx = bool((_ctx or "").strip())
        if have_unified_ctx:
            try:
                # Default ON: machine-level compaction for <embeddings_*> blocks
                cmp_on = str(os.getenv("JINX_CTX_COMPACT", "1")).lower() not in ("", "0", "false", "off", "no")
            except Exception:
                cmp_on = True
            _ctx_final = compact_context(_ctx) if cmp_on else _ctx
            jx = jx + "\n" + _ctx_final + "\n"
        # Auto-inject helpful embedding macros so the user doesn't need to type them (fallback if unified ctx missing)
        try:
            auto_on = str(os.getenv("JINX_AUTOMACROS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            auto_on = True
        if auto_on and (not have_unified_ctx) and ("{{m:" not in jx or "{{m:emb:" not in jx or "{{m:mem:" not in jx):
            lines = await auto_context_lines(txt)
            if lines:
                jx = jx + "\n" + "\n".join(lines) + "\n"
        await _yield0()
        # Optionally include recent patch previews/commits from runtime exports
        try:
            include_patch = str(os.getenv("JINX_AUTOMACRO_PATCH_EXPORTS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            include_patch = True
        if include_patch and ("{{export:" not in jx or "{{export:last_patch_" not in jx):
            exp_lines = await _patch_exports_lines()
            if exp_lines:
                jx = jx + "\n" + "\n".join(exp_lines) + "\n"
        await _yield0()
        # Optionally include last verification results
        try:
            include_verify = str(os.getenv("JINX_AUTOMACRO_VERIFY_EXPORTS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            include_verify = True
        if include_verify and ("{{export:" not in jx or "{{export:last_verify_" not in jx):
            vlines = await _verify_exports_lines()
            if vlines:
                jx = jx + "\n" + "\n".join(vlines) + "\n"
        await _yield0()
        # Optionally include last sandbox run artifacts (stdout/stderr/status) via macros
        try:
            include_run = str(os.getenv("JINX_AUTOMACRO_RUN_EXPORTS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            include_run = True
        if include_run and ("{{m:run:" not in jx):
            rlines = await _run_exports_lines(None)
            if rlines:
                jx = jx + "\n" + "\n".join(rlines) + "\n"
        # Build macro context and expand provider macros {{m:ns:arg1:arg2}}
        try:
            anc = await load_last_anchors()
        except Exception:
            anc = {}
        try:
            progs = await list_programs()
        except Exception:
            progs = []
        await _yield0()
        ctx = MacroContext(
            key=tag,
            anchors={k: [str(x) for x in (anc.get(k) or [])] for k in ("questions","symbols","paths")},
            programs=progs,
            os_name=platform.system(),
            py_ver=sys.version.split(" ")[0],
            cwd=os.getcwd() if hasattr(os, "getcwd") else "",
            now_iso=_dt.datetime.now().isoformat(timespec="seconds"),
            now_epoch=str(int(_dt.datetime.now().timestamp())),
            input_text=txt or "",
        )
        # Ensure built-in providers and plugin macros are registered/loaded
        # Initialize macro providers/plugins once per process
        import asyncio as _asyncio
        _init_lock = getattr(spark_openai, "_macro_init_lock", None)
        if _init_lock is None:
            _init_lock = _asyncio.Lock()
            setattr(spark_openai, "_macro_init_lock", _init_lock)
        if not getattr(spark_openai, "_macro_inited", False):
            async with _init_lock:
                if not getattr(spark_openai, "_macro_inited", False):
                    try:
                        await register_builtin_macros()
                    except Exception:
                        pass
                    try:
                        await load_macro_plugins()
                    except Exception:
                        pass
                    setattr(spark_openai, "_macro_inited", True)
        try:
            max_exp = int(os.getenv("JINX_PROMPT_MACRO_MAX", "50"))
        except Exception:
            max_exp = 50
        jx = await expand_dynamic_macros(jx, ctx, max_expansions=max_exp)
        await _yield0()
        # Best-effort token hint (chars/4 heuristic) for dynamic memory budgets
        try:
            est_tokens = max(0, (len(jx) + len(txt or "")) // 4)
            await write_token_hint(est_tokens)
        except Exception:
            pass
    except Exception:
        pass
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    # Sanitize prompts to avoid leaking internal .jinx paths/content
    sx = sanitize_prompt_for_external_api(jx)
    stxt = sanitize_prompt_for_external_api(txt or "")
    return jx, tag, model, sx, stxt


async def spark_openai(txt: str, *, prompt_override: str | None = None) -> tuple[str, str]:
    """Call OpenAI Responses API and return output text with the code tag.

    Returns (output_text, code_tag_id).
    """
    jx, tag, model, sx, stxt = await _prepare_request(txt, prompt_override=prompt_override)

    async def openai_task() -> tuple[str, str]:
        req_path: str = ""
        import asyncio as _asyncio
        # Overlap request dump with LLM call
        dump_task = _asyncio.create_task(write_openai_request_dump(
            target_dir=OPENAI_REQUESTS_DIR_GENERAL,
            kind="GENERAL",
            instructions=sx,
            input_text=stxt,
            model=model,
        ))
        # Preferred: validated multi-sample path
        try:
            async with timing_section("llm.call"):
                out = await call_openai_validated(sx, model, stxt, code_id=tag)
        except Exception:
            # Fallback to legacy single-sample on error
            async with timing_section("llm.call_legacy"):
                out = await call_openai(sx, model, stxt)
        # Get dump path (await, then append in background)
        try:
            req_path = await dump_task
        except Exception:
            req_path = ""
        try:
            _asyncio.create_task(write_openai_response_append(req_path, "GENERAL", out))
        except Exception:
            pass
        return (out, tag)

    # Avoid duplicate outbound API calls on post-call exceptions by disabling retries here.
    # Lower-level resiliency is provided by caching/coalescing/multi-path logic.
    return await detonate_payload(openai_task, retries=1)


async def spark_openai_streaming(txt: str, *, prompt_override: str | None = None, on_first_block=None) -> tuple[str, str]:
    """Streaming LLM call with early execution on first complete <python_{tag}> block.

    Returns (full_output_text, code_tag_id).
    """
    jx, tag, model, sx, stxt = await _prepare_request(txt, prompt_override=prompt_override)

    async def openai_task() -> tuple[str, str]:
        req_path: str = ""
        import asyncio as _asyncio
        dump_task = _asyncio.create_task(write_openai_request_dump(
            target_dir=OPENAI_REQUESTS_DIR_GENERAL,
            kind="GENERAL",
            instructions=sx,
            input_text=stxt,
            model=model,
        ))
        try:
            async with timing_section("llm.stream"):
                out = await call_openai_stream_first_block(sx, model, stxt, code_id=tag, on_first_block=on_first_block)
        except Exception:
            async with timing_section("llm.call_fallback"):
                out = await call_openai_validated(sx, model, stxt, code_id=tag)
        try:
            req_path = await dump_task
        except Exception:
            req_path = ""
        try:
            _asyncio.create_task(write_openai_response_append(req_path, "GENERAL", out))
        except Exception:
            pass
        return (out, tag)

    return await detonate_payload(openai_task, retries=1)
