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
