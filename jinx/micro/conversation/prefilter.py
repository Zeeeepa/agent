from __future__ import annotations

import re
from typing import Optional

from jinx.micro.conversation.turns_router import detect_turn_query as _fast_turn

# Lightweight, zero-IO prefilters to avoid unnecessary LLM calls in hard RT.

_MEM_TOKENS = [
    # English
    r"\bmemory\b", r"\bremember\b", r"\brecall\b", r"\bfind\b", r"\bsearch\b", r"\bpins?\b",
    # Russian
    r"\bпамя\w*\b", r"\bвспомн\w*\b", r"\bнайд\w*\b", r"\bпоис\w*\b", r"\bпин(?:ы|ов)?\b", r"\bзакреп\w*\b",
]
_MEM_RE = re.compile("|".join(_MEM_TOKENS), re.IGNORECASE)


def likely_turn_query(text: str) -> bool:
    """True if fast detector can extract an index; avoids LLM when not relevant."""
    try:
        ft = _fast_turn(text or "")
        return bool(ft and int(ft.get("index", 0)) > 0)
    except Exception:
        return False


def likely_memory_action(text: str) -> bool:
    """True if query likely asks about memory retrieval or pins; avoids LLM otherwise."""
    t = (text or "").strip()
    if not t:
        return False
    return bool(_MEM_RE.search(t))


__all__ = ["likely_turn_query", "likely_memory_action"]
