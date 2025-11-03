from __future__ import annotations

from typing import Optional, List
import re as _re

from .context_builder import build_project_context_for, build_project_context_multi_for  # noqa: F401
from .memory_context import build_memory_context_for
from .unifier import build_unified_brain_block


def _split_queries(q: str) -> List[str]:
    s = (q or "").strip()
    if not s:
        return []
    # Heuristic: split by line or sentence boundaries; keep non-empty chunks
    parts = [p.strip() for p in _re.split(r"[\r\n]+|(?<=[\.!?])\s+", s) if (p or '').strip()]
    # Bound number of sub-queries to keep RT budget tight
    return parts[:4]


async def build_unified_context_for(query: str, *, max_chars: Optional[int] = None, max_time_ms: Optional[int] = 300) -> str:
    """Return a unified context string composed of <embeddings_*> blocks.

    Strategy (RT-safe with fallbacks):
    - If multiple sub-queries detected, try multi-query builder first.
    - Otherwise, try single-query builder.
    - If empty, fall back to memory-only block (<embeddings_memory>).
    Errors are swallowed and an empty string is returned on failure.
    """
    q = (query or "").strip()
    if not q:
        return ""
    # Prefer multi when user provided several sentences/lines
    body = ""
    subs = _split_queries(q)
    if len(subs) >= 2:
        try:
            body = await build_project_context_multi_for(subs, max_chars=max_chars, max_time_ms=max_time_ms)
        except Exception:
            body = ""
    if not (body or "").strip():
        try:
            body = await build_project_context_for(q, max_chars=max_chars, max_time_ms=max_time_ms)
        except Exception:
            body = ""
    if body and body.strip():
        # Try to augment with memory context (small budget) and a unified brain block
        try:
            mem_lines = await build_memory_context_for(q, max_chars=600, max_time_ms=160)
        except Exception:
            mem_lines = ""
        try:
            brain = build_unified_brain_block(body, mem_lines or "", q, "")
        except Exception:
            brain = ""
        if brain:
            return (body + "\n\n" + brain)
        return body
    # Fallback to memory-only context
    try:
        return await build_memory_context_for(q, max_chars=1200, max_time_ms=220)
    except Exception:
        return ""


__all__ = ["build_unified_context_for"]
