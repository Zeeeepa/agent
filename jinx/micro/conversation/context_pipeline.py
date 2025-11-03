from __future__ import annotations

import asyncio
import os
from typing import Tuple

from jinx.micro.conversation.phases import (
    build_runtime_base_ctx as _phase_base_ctx,
    build_runtime_mem_ctx as _phase_mem_ctx,
    build_project_context_enriched as _phase_proj_ctx,
)
from jinx.micro.rt.activity import set_activity as _act


async def build_contexts(query: str, *, user_text: str = "", synth: str = "") -> Tuple[str, str, str]:
    """Build (base_ctx, mem_ctx, proj_ctx) concurrently under RT budgets.

    - base_ctx: retrieved from runtime embeddings
    - mem_ctx: optional embeddings-backed memory context (env-gated)
    - proj_ctx: enriched project embeddings context
    """
    _act("retrieving runtime context")
    base_task = asyncio.create_task(_phase_base_ctx(query))
    mem_task = None
    try:
        if str(os.getenv("JINX_EMBED_MEMORY_CTX", "0")).lower() not in ("", "0", "false", "off", "no"):
            mem_task = asyncio.create_task(_phase_mem_ctx(query))
    except Exception:
        mem_task = None

    _act("assembling project context")
    proj_task = asyncio.create_task(_phase_proj_ctx(query, user_text=user_text or "", synth=synth or ""))

    # Await in the natural order: base -> mem(optional) -> proj
    try:
        base_ctx = await base_task
    except Exception:
        base_ctx = ""
    mem_ctx = ""
    if mem_task is not None:
        try:
            mem_ctx = await mem_task
        except Exception:
            mem_ctx = ""
    try:
        proj_ctx = await proj_task
    except Exception:
        proj_ctx = ""
    return (base_ctx or "", mem_ctx or "", proj_ctx or "")
