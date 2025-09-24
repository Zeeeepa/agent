from __future__ import annotations

import asyncio
from typing import List

from jinx.net import get_openai_client
from .project_retrieval_config import PROJ_QUERY_MODEL


async def embed_query(text: str) -> List[float]:
    async def _call():
        def _worker():
            client = get_openai_client()
            return client.embeddings.create(model=PROJ_QUERY_MODEL, input=text)
        return await asyncio.to_thread(_worker)

    try:
        resp = await _call()
        return resp.data[0].embedding if getattr(resp, "data", None) else []
    except Exception:
        return []
