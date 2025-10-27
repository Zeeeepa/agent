from __future__ import annotations

import os
from jinx.micro.text.heuristics import is_code_like as _is_code_like


def _is_short_reply(x: str) -> bool:
    t = (x or "").strip()
    if not t:
        return False
    try:
        thr = max(20, int(os.getenv("JINX_CONTINUITY_SHORTLEN", "80")))
    except Exception:
        thr = 80
    if len(t) <= thr and not _is_code_like(t):
        return True
    return False


def is_short_followup(x: str) -> bool:
    """Public check for short clarification replies (language-agnostic)."""
    return _is_short_reply(x)
