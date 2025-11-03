from __future__ import annotations

import asyncio
import time
from typing import Optional

import jinx.state as jx_state


class TimeSlicer:
    """Cooperative time-slice gate to keep the event loop responsive (adaptive).

    Features:
    - Minimal overhead expiration check by wall-time.
    - Optional adaptive slice length driven by event-loop lag EMA and throttle.
    - No external scheduling; simply awaits `asyncio.sleep(0)` when expired.

    Usage:
        ts = TimeSlicer(ms=12, adaptive=True)
        ...
        await ts.maybe_yield()
    """

    def __init__(self, ms: int = 12, *, adaptive: bool = True, min_ms: int = 6, max_ms: int = 40, throttle_boost: float = 1.5) -> None:
        self.ms = max(1, int(ms))
        self.adaptive = bool(adaptive)
        self.min_ms = max(1, int(min_ms))
        self.max_ms = max(self.min_ms, int(max_ms))
        self.throttle_boost = float(throttle_boost)
        self._t0: float = time.perf_counter()

    def _maybe_adapt(self) -> None:
        if not self.adaptive:
            return
        try:
            lag = float(getattr(jx_state, "lag_ema_ms", 0.0) or 0.0)
        except Exception:
            lag = 0.0
        base = self.ms
        # Increase slice when throttle is active (less yielding, more batching)
        try:
            if bool(getattr(jx_state, "throttle_event", None) and jx_state.throttle_event.is_set()):
                base = int(base * self.throttle_boost)
        except Exception:
            pass
        # Smoothly scale with lag up to max_ms (heuristic)
        if lag > 0.0:
            try:
                scale = 1.0 + min(1.5, lag / 120.0)  # at 120ms overrun => up to +150%
                base = int(base * scale)
            except Exception:
                pass
        self.ms = max(self.min_ms, min(self.max_ms, base))

    def expired(self) -> bool:
        return ((time.perf_counter() - self._t0) * 1000.0) >= self.ms

    async def maybe_yield(self) -> bool:
        """Yield if the current slice has expired. Returns True if yielded."""
        if not self.expired():
            return False
        await asyncio.sleep(0)
        # Adapt slice length on yield boundary
        self._maybe_adapt()
        self._t0 = time.perf_counter()
        return True

    def reset(self) -> None:
        self._t0 = time.perf_counter()


__all__ = ["TimeSlicer"]
