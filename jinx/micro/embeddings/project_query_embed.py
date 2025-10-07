from __future__ import annotations

import asyncio
from typing import List

from .project_retrieval_config import PROJ_QUERY_MODEL
from .embed_cache import embed_text_cached


async def embed_query(text: str) -> List[float]:
    try:
        return await embed_text_cached(text, model=PROJ_QUERY_MODEL)
    except Exception:
        return []
