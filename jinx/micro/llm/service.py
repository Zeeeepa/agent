from __future__ import annotations

import os
from jinx.openai_mod import build_header_and_tag, call_openai
from jinx.log_paths import OPENAI_REQUESTS_DIR_GENERAL
from jinx.logger.openai_requests import write_openai_request_dump, write_openai_response_append
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
        # Auto-inject helpful embedding macros so the user doesn't need to type them
        try:
            auto_on = str(os.getenv("JINX_AUTOMACROS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            auto_on = True
        # Heuristic: code-like queries prefer project context; otherwise dialogue
        def _is_codey(s: str) -> bool:
            if not s:
                return False
            ql = s.lower()
            if any(kw in ql for kw in ("def ", "class ", "import ", "from ", "return ", "async ", "await ")):
                return True
            return any(c in s for c in "=[](){}.:,")
        if auto_on and ("{{m:" not in jx or "{{m:emb:" not in jx):
            lines = []
            try:
                use_dlg = str(os.getenv("JINX_AUTOMACRO_DIALOGUE", "1")).lower() not in ("", "0", "false", "off", "no")
            except Exception:
                use_dlg = True
            try:
                use_proj = str(os.getenv("JINX_AUTOMACRO_PROJECT", "1")).lower() not in ("", "0", "false", "off", "no")
            except Exception:
                use_proj = True
            # Dynamic topK per source
            try:
                dlg_k = int(os.getenv("JINX_AUTOMACRO_DIALOGUE_K", "3"))
            except Exception:
                dlg_k = 3
            try:
                proj_k = int(os.getenv("JINX_AUTOMACRO_PROJECT_K", "3"))
            except Exception:
                proj_k = 3
            codey = _is_codey(txt or "")
            # Prefer project for code-like, dialogue for plain text; keep both if allowed
            if use_dlg:
                if codey and not use_proj:
                    lines.append(f"Context (dialogue): {{m:emb:dialogue:{dlg_k}}}")
                elif not codey:
                    lines.append(f"Context (dialogue): {{m:emb:dialogue:{dlg_k}}}")
            if use_proj:
                if codey or not use_dlg:
                    lines.append(f"Context (code): {{m:emb:project:{proj_k}}}")
            if lines:
                jx = jx + "\n" + "\n".join(lines) + "\n"
        # Optionally include recent patch previews/commits from runtime exports
        try:
            include_patch = str(os.getenv("JINX_AUTOMACRO_PATCH_EXPORTS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            include_patch = True
        if include_patch and ("{{export:" not in jx or "{{export:last_patch_" not in jx):
            exp_lines = [
                "Recent Patch Preview (may be empty): {{export:last_patch_preview:1}}",
                "Recent Patch Commit (may be empty): {{export:last_patch_commit:1}}",
                "Recent Patch Strategy: {{export:last_patch_strategy:1}}",
                "Recent Patch Reason: {{export:last_patch_reason:1}}",
            ]
            jx = jx + "\n" + "\n".join(exp_lines) + "\n"
        # Optionally include last verification results
        try:
            include_verify = str(os.getenv("JINX_AUTOMACRO_VERIFY_EXPORTS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            include_verify = True
        if include_verify and ("{{export:" not in jx or "{{export:last_verify_" not in jx):
            vlines = [
                "Verification Score: {{export:last_verify_score:1}}",
                "Verification Reason: {{export:last_verify_reason:1}}",
                "Verification Files: {{export:last_verify_files:1}}",
            ]
            jx = jx + "\n" + "\n".join(vlines) + "\n"
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
        # Ensure built-in providers and plugin macros are registered/loaded
        try:
            await register_builtin_macros()
        except Exception:
            pass
        try:
            await load_macro_plugins()
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
