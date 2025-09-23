from __future__ import annotations

import asyncio
from typing import Any

from jinx.logging_service import bomb_log
from jinx.micro.rag.file_search import build_file_search_tools
from jinx.net import get_openai_client


async def call_openai(instructions: str, model: str, input_text: str) -> str:
    """Call OpenAI Responses API and return output text.

    Uses to_thread to run the sync SDK call and relies on the shared retry helper
    at the caller site to provide resiliency.
    """
    try:
        extra_kwargs: dict[str, Any] = build_file_search_tools()
        def _worker():
            client = get_openai_client()
            return client.responses.create(
                instructions=instructions,
                model=model,
                input=input_text,
                **extra_kwargs,
            )
        r = await asyncio.to_thread(_worker)
        return r.output_text
    except Exception as e:
        await bomb_log(f"ERROR cortex exploded: {e}")
        raise
