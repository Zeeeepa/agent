from __future__ import annotations

from typing import List, Tuple


def find_line_window(text: str, tokens: List[str], around: int = 6) -> Tuple[int, int, str]:
    """Find a small line window around the first occurrence of any token.

    Returns (line_start, line_end, snippet). Lines are 1-based and inclusive.
    If nothing is found, returns (0, 0, "").
    """
    if not text or not tokens:
        return 0, 0, ""
    lowered = text.lower()
    hit_pos = -1
    hit_len = 0
    for t in tokens:
        if not t:
            continue
        p = lowered.find(t.lower())
        if p >= 0 and (hit_pos == -1 or p < hit_pos):
            hit_pos = p
            hit_len = len(t)
    if hit_pos < 0:
        return 0, 0, ""
    pre = text[:hit_pos]
    ls = pre.count("\n") + 1
    le = ls + max(1, text[hit_pos:hit_pos + hit_len].count("\n"))
    lines_all = text.splitlines()
    a = max(1, ls - around)
    b = min(len(lines_all), le + around)
    snippet = "\n".join(lines_all[a - 1:b]).strip()
    return a, b, snippet
