from __future__ import annotations

import os
from jinx.openai_mod import build_header_and_tag, call_openai
from jinx.log_paths import OPENAI_REQUESTS_DIR_GENERAL
from jinx.logger.openai_requests import write_openai_request_dump, write_openai_response_append
from jinx.retry import detonate_payload
from .prompt_compose import compose_dynamic_prompt
from .macro_registry import MacroContext, expand_dynamic_macros
from .macro_providers import register_builtin_macros
from jinx.micro.conversation.cont import load_last_anchors
from jinx.micro.runtime.api import list_programs
import platform
import sys
import datetime as _dt


async def code_primer(prompt_override: str | None = None) -> tuple[str, str]:
    """Build instruction header and return it with a code tag identifier.

    Returns (header_plus_prompt, code_tag_id).
    """
    return await build_header_and_tag(prompt_override)


async def spark_openai(txt: str, *, prompt_override: str | None = None) -> tuple[str, str]:
    """Call OpenAI Responses API and return output text with the code tag.

    Returns (output_text, code_tag_id).
    """
    jx, tag = await code_primer(prompt_override)
    # Expand dynamic prompt macros in real time (vars/env/anchors/sys/runtime/exports + custom providers)
    try:
        jx = await compose_dynamic_prompt(jx, key=tag)
        # Build macro context and expand provider macros {{m:ns:arg1:arg2}}
        try:
            anc = await load_last_anchors()
        except Exception:
            anc = {}
        try:
            progs = await list_programs()
        except Exception:
            progs = []
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
        # Ensure built-in providers are registered
        try:
            await register_builtin_macros()
        except Exception:
            pass
        try:
            max_exp = int(os.getenv("JINX_PROMPT_MACRO_MAX", "50"))
        except Exception:
            max_exp = 50
        jx = await expand_dynamic_macros(jx, ctx, max_expansions=max_exp)
    except Exception:
        pass
    model = os.getenv("OPENAI_MODEL", "gpt-5")

    async def openai_task() -> tuple[str, str]:
        req_path: str = ""
        try:
            req_path = await write_openai_request_dump(
                target_dir=OPENAI_REQUESTS_DIR_GENERAL,
                kind="GENERAL",
                instructions=jx,
                input_text=txt,
                model=model,
            )
        except Exception:
            pass
        out = await call_openai(jx, model, txt)
        try:
            await write_openai_response_append(req_path, "GENERAL", out)
        except Exception:
            pass
        return (out, tag)

    return await detonate_payload(openai_task)
