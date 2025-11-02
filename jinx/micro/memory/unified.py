from __future__ import annotations

import os
from typing import List

from .router import assemble_memroute as _assemble_memroute


async def assemble_unified_memory_lines(query: str, *, k: int = 12, preview_chars: int = 160, max_time_ms: int | None = None) -> List[str]:
    """Unified memory assembly used by macros and <embeddings_memory> builders.

    Delegates to memroute (pins + graph-aligned + vector + kb + ranker) with RT bounds.
    """
    q = (query or "").strip()
    if not q:
        return []
    # Let memroute manage its own internal budget. We pass k and preview clamp.
    try:
        k_eff = max(1, int(k or int(os.getenv("JINX_MACRO_MEM_TOPK", "12"))))
    except Exception:
        k_eff = 12
    try:
        clamp = max(24, int(preview_chars or int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160"))))
    except Exception:
        clamp = 160
    try:
        lines = await _assemble_memroute(q, k=k_eff, preview_chars=clamp)
    except Exception:
        lines = []
    # Dedup and clamp again, just in case
    out: List[str] = []
    seen: set[str] = set()
    for ln in (lines or []):
        s = (ln or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s[:clamp])
        if len(out) >= k_eff:
            break
    return out


__all__ = ["assemble_unified_memory_lines"]
