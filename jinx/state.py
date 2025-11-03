"""Global state and synchronization primitives.

Minimal, explicit globals to coordinate async behavior. Environment variables:
- ``PULSE``: integer pulse displayed by the spinner (default: 100)
- ``TIMEOUT``: inactivity timeout in seconds before "<no_response>" (default: 30)
"""

from __future__ import annotations

import asyncio
import os

# Shared async lock
shard_lock: asyncio.Lock = asyncio.Lock()

# Global mutable state with safe defaults
pulse: int = int(os.getenv("PULSE", "100"))
boom_limit: int = int(os.getenv("TIMEOUT", "30"))

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
