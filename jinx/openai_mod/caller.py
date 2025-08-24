from __future__ import annotations

import os

from jinx.logging_service import bomb_log
from jinx.network_service import get_cortex
from jinx.rag_service import build_file_search_tools


async def call_openai(instructions: str, model: str, input_text: str) -> str:
    """Call OpenAI Responses API and return output text.

    Uses to_thread to run the sync SDK call and relies on the shared retry helper
    at the caller site to provide resiliency.
    """
    try:
        import asyncio

        # Build optional File Search tool kwargs via the micro-module.
        # If OPENAI_VECTOR_STORE_ID is unset/empty, this returns an empty dict.
        extra_kwargs: dict = build_file_search_tools()

        r = await asyncio.to_thread(
            get_cortex().responses.create,
            instructions=instructions,
            model=model,
            input=input_text,
            **extra_kwargs,
        )
        return r.output_text
    except Exception as e:
        await bomb_log(f"ERROR cortex exploded: {e}")
        raise
