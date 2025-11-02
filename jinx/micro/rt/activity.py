from __future__ import annotations

import time
import jinx.state as state


def set_activity(msg: str) -> None:
    try:
        state.activity = (msg or "").strip()
        state.activity_ts = float(time.perf_counter())
    except Exception:
        # Best-effort: do not crash on spinner metadata
        state.activity = (msg or "").strip()
        state.activity_ts = 0.0


def clear_activity() -> None:
    try:
        state.activity = ""
        state.activity_ts = float(time.perf_counter())
    except Exception:
        state.activity = ""
        state.activity_ts = 0.0


def set_activity_detail(detail: dict | None) -> None:
    try:
        state.activity_detail = dict(detail or {}) if detail is not None else None
        state.activity_detail_ts = float(time.perf_counter())
    except Exception:
        state.activity_detail = None
        state.activity_detail_ts = 0.0


def clear_activity_detail() -> None:
    try:
        state.activity_detail = None
        state.activity_detail_ts = float(time.perf_counter())
    except Exception:
        state.activity_detail = None
        state.activity_detail_ts = 0.0
