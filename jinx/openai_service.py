"""OpenAI integration service.

Prepares instruction headers and invokes the OpenAI client with retries.

This module keeps transport concerns minimal and defers resiliency to the
shared retry helper. Consumers receive model output text paired with the
ephemeral code-tag identifier used for parsing executable blocks downstream.
"""

from __future__ import annotations

import os
from jinx.openai_mod import build_header_and_tag, call_openai


async def code_primer() -> tuple[str, str]:
    """Build instruction header and return it with a code tag identifier.

    Returns
    -------
    tuple[str, str]
        ``(header_plus_prompt, code_tag_id)`` where ``code_tag_id`` is used to
        identify code blocks in downstream parsing.
    """
    return await build_header_and_tag()


async def spark_openai(txt: str) -> tuple[str, str]:
    """Call OpenAI Responses API and return output text with the code tag.

    Parameters
    ----------
    txt : str
        Input text for the model.

    Returns
    -------
    tuple[str, str]
        ``(output_text, code_tag_id)``
    """
    jx, tag = await code_primer()
    # Model override via env; otherwise rely on SDK defaults
    model = os.getenv("OPENAI_MODEL", "gpt-5")

    async def openai_task() -> tuple[str, str]:
        out = await call_openai(jx, model, txt)
        return (out, tag)

    from .retry import detonate_payload

    return await detonate_payload(openai_task)
