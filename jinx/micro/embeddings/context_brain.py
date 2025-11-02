from __future__ import annotations

import os
from typing import List, Tuple

from jinx.micro.brain.concepts import activate_concepts as _brain_activate


def brain_enabled(env_key: str = "EMBED_BRAIN_ENABLE") -> bool:
    try:
        return os.getenv(env_key, "1").strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def brain_topk(default_topk: int = 12, env_key: str = "EMBED_BRAIN_TOP_K") -> int:
    try:
        return max(1, int(os.getenv(env_key, str(default_topk))))
    except Exception:
        return max(1, default_topk)


async def brain_pairs_for(query: str, *, default_topk: int = 12) -> List[Tuple[str, float]]:
    if not brain_enabled():
        return []
    tk = brain_topk(default_topk)
    try:
        return await _brain_activate(query, top_k=tk)
    except Exception:
        return []


__all__ = ["brain_enabled", "brain_topk", "brain_pairs_for"]
