from __future__ import annotations

import re
import ast
from typing import Optional

_CODE_FRAG_RE = re.compile(r"[A-Za-z0-9_\./:\-+*<>=!\"'\[\]\(\)\{\),\s]+", re.DOTALL)


def _is_python(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 3:
        return False
    try:
        ast.parse(s, mode="exec")
        return True
    except Exception:
        pass
    try:
        ast.parse(s, mode="eval")
        return True
    except Exception:
        pass
    try:
        ast.parse(f"({s})", mode="eval")
        return True
    except Exception:
        return False


def extract_code_core(query: str) -> Optional[str]:
    """Extract the most plausible Python code fragment from an arbitrary query string.

    Strategy:
    - Find spans of code-like characters.
    - Prefer substrings that parse as Python (exec/eval/(...) eval) and are longest.
    - As a fallback, return the longest code-like span if any.
    """
    q = (query or "").strip()
    if not q:
        return None
    cands = list(_CODE_FRAG_RE.finditer(q))
    best = ""
    best_len = 0
    for m in cands:
        frag = (m.group(0) or "").strip()
        if len(frag) < 6:
            continue
        if _is_python(frag) and len(frag) > best_len:
            best = frag
            best_len = len(frag)
    if best:
        return best
    if cands:
        longest = max((m.group(0) or '').strip() for m in cands if (m.group(0) or '').strip())
        return longest or None
    return None


__all__ = ["extract_code_core"]
