from __future__ import annotations

import asyncio
import os
from typing import Any

from jinx.logging_service import bomb_log
from jinx.micro.rag.file_search import build_file_search_tools
from jinx.net import get_openai_client
from .llm_cache import call_openai_cached


async def call_openai(instructions: str, model: str, input_text: str) -> str:
    """Call OpenAI Responses API and return output text.

    Uses to_thread to run the sync SDK call and relies on the shared retry helper
    at the caller site to provide resiliency.
    """
    try:
        # Auto-adjustment: if no API key is configured, return a graceful stub response
        if not (os.getenv("OPENAI_API_KEY") or ""):
            await bomb_log("OPENAI_API_KEY missing; LLM disabled — returning stub output")
            return (
                "<llm_disabled>\n"
                "No OpenAI API key configured. Set OPENAI_API_KEY in .env to enable model calls.\n"
                "</llm_disabled>"
            )
        extra_kwargs: dict[str, Any] = build_file_search_tools()
        # Use cached/coalesced LLM call to prevent duplicate concurrent requests
        out = await call_openai_cached(
            instructions=instructions,
            model=model,
            input_text=input_text,
            extra_kwargs=extra_kwargs,
        )
        return out
    except Exception as e:
        await bomb_log(f"ERROR cortex exploded: {e}")
        raise
