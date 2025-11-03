from __future__ import annotations

from typing import Optional

def clamp_int(val: int, lo: int, hi: int) -> int:
    lo = int(lo)
    hi = int(hi)
    if lo > hi:
        lo, hi = hi, lo
    try:
        v = int(val)
    except Exception:
        v = lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def clamp_float(val: float, lo: float, hi: float) -> float:
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        lo, hi = hi, lo
    try:
        v = float(val)
    except Exception:
        v = lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v
