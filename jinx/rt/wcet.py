from __future__ import annotations

"""
WCET Estimator (best-effort): tracks moving averages and p95-like estimates
for operation classes and provides deadline hints.
"""

import threading
import time
from typing import Dict


class _EMA:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.mean_ms = 0.0
        self.sq_mean = 0.0
        self.n = 0

    def update(self, x_ms: float) -> None:
        a = self.alpha
        self.mean_ms = a * x_ms + (1 - a) * self.mean_ms if self.n > 0 else x_ms
        self.sq_mean = a * (x_ms * x_ms) + (1 - a) * self.sq_mean if self.n > 0 else (x_ms * x_ms)
        self.n += 1

    def p95(self) -> float:
        # Rough p95 via mean + 1.64*std (Welford approx from EMA moments)
        if self.n <= 1:
            return self.mean_ms * 2.0 if self.mean_ms > 0 else 1000.0
        var = max(0.0, self.sq_mean - self.mean_ms * self.mean_ms)
        std = var ** 0.5
        return max(self.mean_ms, self.mean_ms + 1.64 * std)


_lock = threading.RLock()
_stats: Dict[str, _EMA] = {}


def update(op: str, dt_ms: float) -> None:
    with _lock:
        s = _stats.get(op)
        if s is None:
            s = _EMA()
            _stats[op] = s
        s.update(max(0.0, float(dt_ms)))


def estimate_deadline_ms(op: str, base_ms: int) -> int:
    with _lock:
        s = _stats.get(op)
        if s is None:
            return int(base_ms)
        # Add margin factor to p95
        m = s.p95() * 1.25
        # Clamp between 50ms and 60s
        return int(max(50.0, min(60000.0, m)))
