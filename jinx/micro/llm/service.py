from __future__ import annotations

import os
from jinx.openai_mod import build_header_and_tag, call_openai
from jinx.log_paths import OPENAI_REQUESTS_DIR_GENERAL
from jinx.logger.openai_requests import write_openai_request_dump, write_openai_response_append
from jinx.retry import detonate_payload


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
