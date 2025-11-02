from __future__ import annotations

import re
from typing import List, Tuple

# Unicode-safe identifier: starts with letter/underscore then word chars
NAME = r"(?:[^\W\d]\w*)"
NAME_RE = re.compile(NAME)


def is_camel_case(tok: str) -> bool:
    s = tok or ""
    if len(s) <= 3:
        return False
    has_upper = any(ch.isupper() for ch in s[1:])
    has_lower = any(ch.islower() for ch in s)
    return has_upper and has_lower


def is_pathlike(x: str) -> bool:
    s = x or ""
    return ("/" in s or "\\" in s) and (" " not in s)


def match_paren(s: str, i_open: int) -> int:
    depth = 0
    for i in range(i_open, len(s)):
        ch = s[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def match_bracket(s: str, i_open: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(i_open, len(s)):
        ch = s[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_args(body: str) -> List[str]:
    args: List[str] = []
    cur = []
    depth = 0
    for ch in body or "":
        if ch == '(':
            depth += 1
            cur.append(ch)
            continue
        if ch == ')':
            cur.append(ch)
            depth = max(0, depth - 1)
            continue
        if ch == ',' and depth == 0:
            part = "".join(cur).strip()
            if part:
                args.append(part)
            cur = []
            continue
        cur.append(ch)
    part = "".join(cur).strip()
    if part:
        args.append(part)
    return [a for a in args if a]
