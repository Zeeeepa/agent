from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from jinx.micro.rt.rt_budget import run_bounded as _bounded, env_ms as _env_ms
from jinx.micro.rt.activity import set_activity_detail as _actdet
from jinx.micro.common.log import log_info as _log_info


@dataclass
class StageResult:
    ok: bool
    value: Any | None
    error: str | None
    timed_out: bool
    dur_ms: int


async def run_stage(
    name: str,
    awaitable_or_factory: Awaitable[Any] | Callable[[], Awaitable[Any]],
    *,
    timeout_env: str,
    default_ms: int,
    activity_stage: Optional[str] = None,
) -> StageResult:
    """Run a single stage with a hard RT budget and basic telemetry.

    - name: stage name for logs
    - awaitable_or_factory: coroutine or callable returning a coroutine
    - timeout_env/default_ms: env key + default budget
    - activity_stage: spinner activity detail stage identifier
    """
    t0 = time.perf_counter()
    ms = _env_ms(timeout_env, default_ms)
    try:
        if activity_stage:
            try:
                _actdet({"stage": activity_stage, "rem_ms": ms})
            except Exception:
                pass
        aw = awaitable_or_factory() if callable(awaitable_or_factory) else awaitable_or_factory
        val = await _bounded(aw, ms)
        dur = int((time.perf_counter() - t0) * 1000.0)
        if val is None and ms > 0:
            try:
                _log_info("stage.timeout", name=name, ms=ms, dur=dur)
            except Exception:
                pass
            return StageResult(False, None, None, True, dur)
        try:
            _log_info("stage.ok", name=name, ms=ms, dur=dur)
        except Exception:
            pass
        return StageResult(True, val, None, False, dur)
    except Exception as e:
        dur = int((time.perf_counter() - t0) * 1000.0)
        try:
            _log_info("stage.error", name=name, ms=ms, dur=dur)
        except Exception:
            pass
        return StageResult(False, None, str(e), False, dur)
