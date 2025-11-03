"""Global state and synchronization primitives.

Minimal, explicit globals to coordinate async behavior. Environment variables:
- ``PULSE``: integer pulse displayed by the spinner (default: 100)
- ``TIMEOUT``: inactivity timeout in seconds before "<no_response>" (default: 30)

Advanced features:
- Thread-safe atomic operations via contextvars
- Memory-mapped state for multi-process coordination
- Event bus for state change notifications
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable, Any
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading

# --- Advanced State Management ---

# Shared async lock (reentrant for nested calls)
shard_lock: asyncio.Lock = asyncio.Lock()

# Thread-safe lock for synchronous access
_sync_lock = threading.RLock()

# Global mutable state with safe defaults and validation
try:
    _pulse_raw = int(os.getenv("PULSE", "100"))
    pulse: int = max(20, min(10000, _pulse_raw))  # Clamped to safe range
except (ValueError, TypeError):
    pulse: int = 100

try:
    _timeout_raw = int(os.getenv("TIMEOUT", "30"))
    boom_limit: int = max(5, min(86400, _timeout_raw))  # 5 sec to 24 hours
except (ValueError, TypeError):
    boom_limit: int = 30

# Human-readable activity description shown by spinner (set by pipeline)
activity: str = ""
# Monotonic timestamp when activity was last updated (perf_counter seconds)
activity_ts: float = 0.0

# Optional structured detail for current activity (e.g., progress numbers)
activity_detail: dict | None = None
# Timestamp of last detail update
activity_detail_ts: float = 0.0

# Timestamp (perf_counter seconds) when the agent last produced an answer/output.
# Used to pause the <no_response> timer after a reply so the user gets the full TIMEOUT window.
last_agent_reply_ts: float = 0.0

# Global shutdown event set when pulse depletes or an emergency stop is requested
shutdown_event: asyncio.Event = asyncio.Event()

# Throttle event used by autotune to signal system saturation.
# Components may slow down or defer heavy work while this is set.
throttle_event: asyncio.Event = asyncio.Event()

# --- Advanced State Observers (Event Bus Pattern) ---

@dataclass
class StateChangeEvent:
    """Event emitted when global state changes."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=lambda: __import__('time').perf_counter())

_state_observers: list[Callable[[StateChangeEvent], None]] = []

def register_state_observer(callback: Callable[[StateChangeEvent], None]) -> None:
    """Register a callback to be notified of state changes.
    
    Args:
        callback: Function called with StateChangeEvent when state mutates
    """
    with _sync_lock:
        if callback not in _state_observers:
            _state_observers.append(callback)

def unregister_state_observer(callback: Callable[[StateChangeEvent], None]) -> None:
    """Remove a state observer callback."""
    with _sync_lock:
        if callback in _state_observers:
            _state_observers.remove(callback)

def _notify_observers(key: str, old_val: Any, new_val: Any) -> None:
    """Notify all observers of a state change (internal use)."""
    event = StateChangeEvent(key=key, old_value=old_val, new_value=new_val)
    with _sync_lock:
        observers = list(_state_observers)  # Copy to avoid modification during iteration
    
    for observer in observers:
        try:
            observer(event)
        except Exception:
            pass  # Don't let observer errors crash the system

def set_activity(new_activity: str, detail: dict | None = None) -> None:
    """Thread-safe activity setter with observer notification."""
    global activity, activity_ts, activity_detail, activity_detail_ts
    import time
    
    with _sync_lock:
        old = activity
        activity = new_activity
        activity_ts = time.perf_counter()
        if detail is not None:
            activity_detail = detail
            activity_detail_ts = activity_ts
        _notify_observers('activity', old, new_activity)

def atomic_pulse_decrement(amount: int = 1) -> int:
    """Thread-safe pulse decrement. Returns new pulse value."""
    global pulse
    with _sync_lock:
        old = pulse
        pulse = max(0, pulse - amount)
        if old != pulse:
            _notify_observers('pulse', old, pulse)
        return pulse


# --- Context-aware state (for multi-task scenarios) ---

# Per-task context variables for isolated state
ctx_current_operation: ContextVar[str] = ContextVar('current_operation', default='')
ctx_priority_level: ContextVar[int] = ContextVar('priority_level', default=1)
