from __future__ import annotations

import re
from typing import Optional

from jinx.micro.conversation.turns_router import detect_turn_query as _fast_turn
from jinx.micro.conversation.mem_intent import likely_memory_action as _mem_likely

# Lightweight, zero-IO prefilters to avoid unnecessary LLM calls in hard RT.


def likely_turn_query(text: str) -> bool:
    """True if fast detector can extract an index; avoids LLM when not relevant."""
    try:
        ft = _fast_turn(text or "")
        return bool(ft and int(ft.get("index", 0)) > 0)
    except Exception:
        return False


def likely_memory_action(text: str) -> bool:
    """True if query likely asks about memory retrieval or pins; language-agnostic.

    Thin facade over `jinx.micro.conversation.mem_intent.likely_memory_action` (char-grams + structural signals).
    """
    return _mem_likely(text or "")


__all__ = ["likely_turn_query", "likely_memory_action"]
