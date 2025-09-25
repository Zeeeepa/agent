from __future__ import annotations

import re
from typing import List

from .project_identifiers import extract_identifiers


def expand_strong_tokens(q: str, max_items: int = 32) -> List[str]:
    """Extract and expand strong code-like tokens from a query.

    - Starts from identifier-like tokens
    - Adds suffix after '.' (e.g., 'self.generic_visit' -> 'generic_visit') when length >= 4
    - Adds long underscore parts (>=6 chars)
    - Returns at most max_items, sorted by length desc, preferring tokens with '_' or '.' or length >= 6
    """
    base = sorted(extract_identifiers(q, max_items=max_items), key=len, reverse=True)
    tok_set: set[str] = set()
    for t in base:
        tl = (t or "").strip()
        if not tl:
            continue
        tok_set.add(tl)
        if "." in tl:
            suf = tl.split(".")[-1]
            if len(suf) >= 4:
                tok_set.add(suf)
        if "_" in tl:
            for p in tl.split("_"):
                if len(p) >= 6:
                    tok_set.add(p)
        # Handle bracketless generics like 'QueueT' by keeping head part
        # Example: 'QueueT' -> 'Queue'
        if len(tl) >= 4 and tl[-1].isupper() and tl[:-1].isalpha():
            head = tl[:-1]
            if len(head) >= 4:
                tok_set.add(head)
        # Split CamelCase into components and add longest meaningful ones
        # e.g., 'AsyncQueueItem' -> 'Async', 'Queue', 'Item'
        parts = []
        try:
            parts = [p for p in __import__('re').split(r'(?<!^)(?=[A-Z])', tl) if p]
        except Exception:
            parts = []
        for p in parts:
            if len(p) >= 4:
                tok_set.add(p)
    toks = [t for t in sorted(tok_set, key=len, reverse=True) if ("_" in t) or ("." in t) or (len(t) >= 6)]
    return toks[:max_items]


def codeish_tokens(q: str) -> List[str]:
    """Extract simpler code-like substrings (alnum/_/.) with length >= 3.

    Intended to complement expand_strong_tokens when scanning raw text.
    """
    toks: list[str] = []
    for m in re.finditer(r"(?u)[\w\.]{3,}", q or ""):
        s = (m.group(0) or "").strip()
        if not s:
            continue
        toks.append(s)
    # Deduplicate preserving order
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out
