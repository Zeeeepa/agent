from __future__ import annotations

import asyncio
from typing import List


async def gather_context_for_subs(subs: List[str], dialog_ms: int, proj_ms: int) -> List[str]:
    """Gather dialogue and project code context blocks for each sub-query.

    Imports the heavy modules inside the function to avoid import-time overheads and coupling.
    """
    parts: List[str] = []
    if not subs:
        return parts
    # Local imports to keep this module light
    from jinx.micro.embeddings.retrieval import build_context_for as build_dialogue_ctx
    from jinx.micro.embeddings.project_retrieval import build_project_context_for as build_project_ctx
    from jinx.micro.conversation.cont import load_last_anchors as _load_last_anchors

    # Continuity: pull small anchor hints (symbols/paths) to stabilize retrieval for each sub-query
    try:
        _anc = await _load_last_anchors()
        _sym = (_anc.get("symbols") or [])[:2]
        _pth = (_anc.get("paths") or [])[:1]
        _hints = " ".join([x for x in (_sym + _pth) if x])
    except Exception:
        _hints = ""

    for i, q in enumerate(subs):
        if i:
            await asyncio.sleep(0)
        try:
            qd = (q + " " + _hints).strip() if _hints else q
            dctx = await build_dialogue_ctx(qd, max_time_ms=dialog_ms)
        except Exception:
            dctx = ""
        try:
            qp = (q + " " + _hints).strip() if _hints else q
            pctx = await build_project_ctx(qp, max_time_ms=proj_ms)
        except Exception:
            pctx = ""
        if dctx:
            parts.append(dctx)
        if pctx:
            parts.append(pctx)
    return parts
