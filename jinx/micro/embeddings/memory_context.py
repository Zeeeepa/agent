from __future__ import annotations

import hashlib
import os
from typing import List

from jinx.micro.memory.unified import assemble_unified_memory_lines as _mem_unified


async def build_memory_context_for(query: str, *, k: int | None = None, max_chars: int = 1500, max_time_ms: int | None = 220) -> str:
    """Build <embeddings_memory> using the unified memory assembler (pins+graph+vector+kb+ranker).

    This unifies the memory shown via macros ({{m:memroute}}) and embeddings context to avoid duplication.
    """
    q = (query or "").strip()
    if not q:
        return ""
    try:
        k_eff = int(k) if k is not None else int(os.getenv("EMBED_TOP_K", "5"))
    except Exception:
        k_eff = 5
    k_eff = max(1, k_eff)
    try:
        clamp = max(24, int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
    except Exception:
        clamp = 160
    try:
        lines = await _mem_unified(q, k=k_eff, preview_chars=clamp)
    except Exception:
        lines = []
    if not lines:
        return ""
    parts: List[str] = []
    total = 0
    for ln in lines:
        L = len(ln) + 1
        if total + L > max_chars:
            break
        parts.append(ln)
        total += L
    if not parts:
        return ""
    body = "\n".join(parts)
    return f"<embeddings_memory>\n{body}\n</embeddings_memory>"


__all__ = ["build_memory_context_for"]
