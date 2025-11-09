from __future__ import annotations

"""
Lightweight admission control for RT constraints.

Provides best-effort concurrency guards per operation class without blocking the
event loop for long. Use guard(op, timeout_ms) as an async context manager.

Operation classes (suggested):
- 'graph'  : code graph scans, tree-sitter, AST-heavy operations
- 'patch'  : autopatch commit/verify phases
- 'llm'    : outbound OpenAI requests (optional gating)
- 'turn'   : per-turn processing (optional, typically handled elsewhere)
"""

import asyncio
import os
from typing import Dict
from contextlib import asynccontextmanager


_sems: Dict[str, asyncio.Semaphore] = {}
_lock = asyncio.Lock()


def _cap_for(op: str) -> int:
    key = f"JINX_ADM_{op.upper()}_CONC"
    try:
        v = int(os.getenv(key, "0"))
        if v > 0:
            return v
    except Exception:
        pass
    # Defaults
    if op == "graph":
        return 1
    if op == "patch":
        return 2
    if op == "llm":
        return 2
    if op == "turn":
        return 4
    return 2


async def _get_sem(op: str) -> asyncio.Semaphore:
    global _sems
    async with _lock:
        sem = _sems.get(op)
        if sem is None:
            sem = asyncio.Semaphore(_cap_for(op))
            _sems[op] = sem
        return sem


@asynccontextmanager
async def guard(op: str, timeout_ms: int = 200):
    """Async context manager; yields True if admitted, False otherwise."""
    sem = await _get_sem(op)
    admitted = False
    try:
        try:
            admitted = await asyncio.wait_for(sem.acquire(), timeout=max(0.01, timeout_ms / 1000.0))
        except asyncio.TimeoutError:
            admitted = False
        except Exception:
            admitted = False
        yield admitted
    finally:
        try:
            if admitted:
                sem.release()
        except Exception:
            pass
