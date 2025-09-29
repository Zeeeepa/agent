from __future__ import annotations

import os
from typing import Optional, List

from jinx.micro.verify.verifier import ensure_verifier_running as _ensure_verifier, submit_verify_embedding as _verify_embed
from jinx.micro.conversation.cont import load_last_anchors as _load_anchors


async def last_goal() -> str:
    try:
        anc = await _load_anchors()
    except Exception:
        anc = {}
    try:
        q = (anc.get("questions") or [])
        if q:
            return str(q[-1]).strip()
    except Exception:
        pass
    return ""


async def maybe_verify(goal: Optional[str], files: List[str], diff: str) -> None:
    try:
        on = str(os.getenv("JINX_VERIFY_AUTORUN", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        on = True
    if not on:
        return
    try:
        await _ensure_verifier()
    except Exception:
        pass
    g = goal or await last_goal()
    if not g:
        return
    try:
        await _verify_embed(g, files=list(files or []), diff=str(diff or ""))
    except Exception:
        pass
