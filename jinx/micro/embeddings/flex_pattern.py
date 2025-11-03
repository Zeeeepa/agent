from __future__ import annotations

import re
from typing import Optional

from .project_query_core import extract_code_core

_WS = re.compile(r"\s+", re.MULTILINE)


def make_flex_code_pattern(src: str, *, ignore_case: bool = False) -> Optional[re.Pattern[str]]:
    """Build a whitespace/punctuation-flex tolerant regex for code-like fragments.

    - Collapses whitespace to single spaces then expands spaces to '\s+'.
    - Allows optional whitespace around '.', '(', ')', ','.
    - Returns compiled pattern with DOTALL and optional IGNORECASE.
    """
    s = (src or "").strip()
    if len(s) < 3:
        return None
    try:
        s = _WS.sub(" ", s)
        esc = re.escape(s)
        # Collapse spaces to flexible whitespace
        esc = esc.replace(r"\ ", r"\s+")
        # Allow flexible spacing around common punctuation
        esc = esc.replace(r"\.", r"\s*\.\s*")
        esc = esc.replace(r"\(", r"\s*\(\s*")
        esc = esc.replace(r"\)", r"\s*\)\s*")
        esc = esc.replace(r"\,", r"\s*,\s*")
        # Allow flexible spacing around '=' and optional type annotation before '='
        # First, allow spaces around '='
        esc = esc.replace(r"\=", r"\s*=\s*")
        # Then, upgrade to accept optional ': type' annotation before '=' (single-line, capped length)
        esc = esc.replace(r"\s*=\s*", r"(?:\s*:\s*[^=\n]{0,80}\s*)?\s*=\s*")
        flags = re.DOTALL | (re.IGNORECASE if ignore_case else 0)
        return re.compile(esc, flags)
    except Exception:
        return None


def make_flex_code_pattern_from_query(query: str, *, prefer_core: bool = True, ignore_case: bool = False) -> Optional[re.Pattern[str]]:
    q = (query or "").strip()
    if not q:
        return None
    src = extract_code_core(q) if prefer_core else q
    src = src or q
    return make_flex_code_pattern(src, ignore_case=ignore_case)


__all__ = [
    "make_flex_code_pattern",
    "make_flex_code_pattern_from_query",
]
