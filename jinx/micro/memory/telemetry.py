from __future__ import annotations

import os
import json
import time
import asyncio
from typing import Any, Dict

from jinx.micro.memory.storage import memory_dir


_METRICS_FILE = os.path.join(memory_dir(), "metrics.jsonl")


def _append_line_sync(path: str, line: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


async def log_metric(kind: str, payload: Dict[str, Any]) -> None:
    """Append a single JSON metric line. Best-effort, non-blocking.

    This function never raises.
    """
    obj = {"ts_ms": int(time.time() * 1000), "kind": kind, **(payload or {})}
    try:
        line = json.dumps(obj, ensure_ascii=False)
    except Exception:
        # Fallback minimal serialization
        line = json.dumps({"ts_ms": obj.get("ts_ms"), "kind": kind})
    try:
        await asyncio.to_thread(_append_line_sync, _METRICS_FILE, line)
    except Exception:
        pass


async def log_memroute_event(stage: str, count: int, elapsed_ms: float, k: int, max_ms: float) -> None:
    """Convenience wrapper for memroute stage metrics."""
    await log_metric("memroute_stage", {
        "stage": str(stage or ""),
        "added": int(max(0, int(count or 0))),
        "elapsed_ms": float(max(0.0, elapsed_ms)),
        "k": int(max(0, k)),
        "max_ms": float(max(0.0, max_ms)),
    })
