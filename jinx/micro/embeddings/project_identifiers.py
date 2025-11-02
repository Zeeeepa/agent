from __future__ import annotations

import re
from typing import List
from jinx.micro.text.structural import is_camel_case as _is_camel_case

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
        if ("_" in t) or ("." in t) or _is_camel_case(t):
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                out.append(t)
                if len(out) >= max_items:
                    break
    return out
