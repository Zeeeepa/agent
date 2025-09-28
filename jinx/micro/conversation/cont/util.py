from __future__ import annotations

import os
from jinx.micro.llm.chain_gate import _CODEY_RE  # reuse same heuristic


def _is_short_reply(x: str) -> bool:
    t = (x or "").strip()
    if not t:
        return False
    try:
        thr = max(20, int(os.getenv("JINX_CONTINUITY_SHORTLEN", "80")))
    except Exception:
        thr = 80
    if len(t) <= thr and not _CODEY_RE.search(t.lower()):
        return True
    return False


def is_short_followup(x: str) -> bool:
    """Public check for short clarification replies (language-agnostic)."""
    return _is_short_reply(x)
