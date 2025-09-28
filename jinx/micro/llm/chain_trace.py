from __future__ import annotations

import json
import time
from typing import Any, Dict

from jinx.log_paths import PLAN_TRACE
from jinx.logger.file_logger import append_line as _append
from .chain_utils import truthy_env


async def trace_plan(payload: Dict[str, Any]) -> None:
    """Append a small JSON line to the planner trace when enabled.

    Controlled by env JINX_CHAINED_TRACE. Writes only aggregate/meta fields
    to avoid leaking sensitive content.
    """
    if not truthy_env("JINX_CHAINED_TRACE", "0"):
        return
    try:
        rec = dict(payload)
        rec["ts"] = int(time.time() * 1000)
        await _append(PLAN_TRACE, json.dumps(rec, ensure_ascii=False))
    except Exception:
        # best-effort
        return
