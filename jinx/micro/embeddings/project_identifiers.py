from __future__ import annotations

import re
from typing import List

# Very light identifier extractor, language-agnostic
# Picks tokens likely to be identifiers (underscored, camelCase, dotted) with length >= 4

_ident_re = re.compile(r"(?u)[\w\.]+")


def extract_identifiers(text: str, max_items: int = 50) -> List[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for m in _ident_re.finditer(text):
        t = m.group(0)
        if len(t) < 4:
            continue
        if t.isdigit():
            continue
        # Heuristics: underscore or dot or camelCase
        if ("_" in t) or ("." in t) or _looks_camel(t):
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                out.append(t)
                if len(out) >= max_items:
                    break
    return out


def _looks_camel(tok: str) -> bool:
    # contains an uppercase letter after the first position or mixed case pattern
    if len(tok) <= 3:
        return False
    has_upper = any(ch.isupper() for ch in tok[1:])
    has_lower = any(ch.islower() for ch in tok)
    return has_upper and has_lower
