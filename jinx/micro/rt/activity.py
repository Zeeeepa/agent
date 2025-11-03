"""Advanced activity tracking with observer pattern integration."""

from __future__ import annotations

import time
import jinx.state as state
from typing import Optional


def set_activity(msg: str, detail: Optional[dict] = None) -> None:
    """Set activity with optional detail, using thread-safe observer pattern.
    
    This function now leverages the improved state management from state.py
    which provides observer notifications and thread safety.
    """
    try:
        # Use the new thread-safe setter from state.py if available
        if hasattr(state, 'set_activity'):
            state.set_activity(msg, detail)
        else:
            # Fallback to direct assignment for backward compatibility
            state.activity = (msg or "").strip()
            state.activity_ts = float(time.perf_counter())
            if detail is not None:
                state.activity_detail = detail
                state.activity_detail_ts = state.activity_ts
    except Exception:
        # Best-effort: do not crash on spinner metadata
        try:
            state.activity = (msg or "").strip()
            state.activity_ts = 0.0
        except Exception:
            pass


def clear_activity() -> None:
    try:
        state.activity = ""
        state.activity_ts = float(time.perf_counter())
    except Exception:
        state.activity = ""
        state.activity_ts = 0.0


def set_activity_detail(detail: dict | None) -> None:
    """Set activity detail with validation."""
    try:
        # Validate detail is actually a dict
        if detail is not None and not isinstance(detail, dict):
            detail = None
        
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
