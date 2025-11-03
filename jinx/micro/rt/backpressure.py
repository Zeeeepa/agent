from __future__ import annotations

import time
from typing import Optional

import jinx.state as jx_state


def set_throttle_ttl(seconds: float = 0.35) -> None:
    """Raise throttle immediately and set a TTL to auto-clear after `seconds`."""
    try:
        jx_state.throttle_event.set()
        jx_state.throttle_unset_ts = float(time.perf_counter()) + max(0.0, seconds)
    except Exception:
        pass


def clear_throttle_if_ttl(now: Optional[float] = None) -> None:
    """Clear throttle when TTL has expired; no-op otherwise."""
    try:
        tz = float(getattr(jx_state, "throttle_unset_ts", 0.0) or 0.0)
    except Exception:
        tz = 0.0
    if not tz:
        return
    try:
        cur = float(now if now is not None else time.perf_counter())
    except Exception:
        cur = time.perf_counter()
    if cur >= tz:
        try:
            jx_state.throttle_event.clear()
            jx_state.throttle_unset_ts = 0.0
        except Exception:
            pass


def update_from_lag(lag_ms: float, *, hi_ms: float = 120.0, lo_ms: float = 50.0) -> None:
    """Sustained hysteresis using EMA lag and counters stored in jx_state.

    - Increment hi counter when lag > hi_ms, reset lo counter.
    - Increment lo counter when lag < lo_ms, reset hi counter.
    - Decay counters by 1 in the mid-band to avoid stickiness.
    - Engage throttle when hi counter >= 4.
    - Clear throttle (if no TTL) when lo counter >= 8.
    """
    try:
        lag = float(lag_ms)
        hi = float(hi_ms)
        lo = float(lo_ms)
    except Exception:
        return
    # Load existing counters from state
    try:
        hi_cnt = int(getattr(jx_state, "_lag_hi_cnt", 0) or 0)
        lo_cnt = int(getattr(jx_state, "_lag_lo_cnt", 0) or 0)
    except Exception:
        hi_cnt = 0; lo_cnt = 0
    try:
        tz = float(getattr(jx_state, "throttle_unset_ts", 0.0) or 0.0)
    except Exception:
        tz = 0.0
    # Update counters
    try:
        if lag > hi:
            hi_cnt += 1; lo_cnt = 0
        elif lag < lo:
            lo_cnt += 1; hi_cnt = 0
        else:
            # Mid-band: decay toward zero slowly
            hi_cnt = max(0, hi_cnt - 1)
            lo_cnt = max(0, lo_cnt - 1)
        setattr(jx_state, "_lag_hi_cnt", hi_cnt)
        setattr(jx_state, "_lag_lo_cnt", lo_cnt)
        # Apply gating with sustained thresholds
        if hi_cnt >= 4:
            jx_state.throttle_event.set()
        elif lo_cnt >= 8 and not tz:
            jx_state.throttle_event.clear()
    except Exception:
        pass


__all__ = [
    "set_throttle_ttl",
    "clear_throttle_if_ttl",
    "update_from_lag",
]
