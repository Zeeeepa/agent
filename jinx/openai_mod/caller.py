from __future__ import annotations

import os

from jinx.logging_service import bomb_log
from jinx.network_service import get_cortex


async def call_openai(instructions: str, model: str, input_text: str) -> str:
    """Call OpenAI Responses API and return output text.

    Uses to_thread to run the sync SDK call and relies on the shared retry helper
    at the caller site to provide resiliency.
    """
    try:
        import asyncio

        r = await asyncio.to_thread(
            get_cortex().responses.create,
            instructions=instructions,
            model=model,
            input=input_text,
        )
        return r.output_text
    except Exception as e:
        await bomb_log(f"ERROR cortex exploded: {e}")
        raise
