from __future__ import annotations

# Thin facade: delegate to micro-module implementation to keep API stable.
from jinx.micro.llm.openai_caller import call_openai as _call_openai


async def call_openai(instructions: str, model: str, input_text: str) -> str:
    """Call OpenAI Responses API and return output text.

    Delegates to ``jinx.micro.llm.openai_caller.call_openai``.
    """
    return await _call_openai(instructions, model, input_text)
