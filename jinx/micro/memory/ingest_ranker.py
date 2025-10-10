from __future__ import annotations

import os
import time
from typing import List, Tuple

from jinx.micro.text.heuristics import is_code_like as _is_code_like
from jinx.micro.embeddings.text_clean import is_noise_text as _is_noise


def _now_ms() -> int:
    try:
        import time as _t
        return int(_t.time() * 1000)
    except Exception:
        return 0


def _lines(txt: str) -> List[str]:
    return [ln.strip() for ln in (txt or "").splitlines() if (ln or "").strip()]


def _score_line(s: str, *, pos_weight: float = 0.0, source_bias: float = 0.0, minlen: int = 12) -> float:
    if not s or len(s) < minlen:
        return -1.0
    if _is_noise(s):
        return -1.0
    score = 0.0
    # length richness (soft cap)
    L = len(s)
    score += min(120, L) / 120.0
    # code-like weight
    try:
        if _is_code_like(s):
            score += 1.0
    except Exception:
        pass
    # punctuation richness (prefer structured lines)
    pch = 0
    for ch in "()[]{}=._:":
        if ch in s:
            pch += 1
    score += min(4, pch) * 0.15
    # bias by where it came from (recency for compact tail, stability for evergreen head)
    score += float(source_bias)
    # positional weight (recency or head priority)
    score += float(pos_weight)
    return score


def build_candidates(compact: str | None, evergreen: str | None, *, k_total: int, minlen: int, include_evergreen: bool) -> List[str]:
    """Select top-k salient memory lines from compact/evergreen with light scoring.

    - Pulls more from compact tail (recency) and a bit from evergreen head (stability).
    - Uses code-like and length features; filters noise.
    - Returns unique lines, best-first.
    """
    k_total = max(1, int(k_total))
    minlen = max(8, int(minlen))

    c_lines = _lines(compact or "")
    e_lines = _lines(evergreen or "") if include_evergreen else []

    # Pre-trim candidates to bound work
    try:
        c_take = max(k_total * 4, 24)
        e_take = max(k_total * 2, 12) if include_evergreen else 0
    except Exception:
        c_take, e_take = k_total * 4, (k_total * 2 if include_evergreen else 0)

    cand: List[Tuple[float, str]] = []

    # Compact tail: strong recency bias
    tail = c_lines[-c_take:]
    n = len(tail)
    for i, s in enumerate(tail):
        # newer lines get higher pos weight
        pos_w = 0.6 * (i / max(1, n - 1)) if n > 1 else 0.6
        sc = _score_line(s, pos_weight=pos_w, source_bias=0.4, minlen=minlen)
        if sc >= 0:
            cand.append((sc, s))

    # Evergreen head + tail: small stability component
    if include_evergreen and e_take > 0 and e_lines:
        head = e_lines[: (e_take // 2)]
        tail_e = e_lines[-(e_take // 2):]
        for i, s in enumerate(head):
            pos_w = 0.4 * (1.0 - (i / max(1, len(head))))
            sc = _score_line(s, pos_weight=pos_w, source_bias=0.2, minlen=minlen)
            if sc >= 0:
                cand.append((sc, s))
        for i, s in enumerate(tail_e):
            pos_w = 0.3 * (i / max(1, len(tail_e)))
            sc = _score_line(s, pos_weight=pos_w, source_bias=0.15, minlen=minlen)
            if sc >= 0:
                cand.append((sc, s))

    # Sort and de-duplicate preserving best-first
    cand.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    out: List[str] = []
    for _sc, s in cand:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= k_total:
            break
    return out


__all__ = ["build_candidates"]
