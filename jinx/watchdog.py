from __future__ import annotations

import asyncio
from typing import Optional

import jinx.state as jx_state
from jinx.settings import Settings


def start_watchdog_task(settings: Settings) -> "asyncio.Task[None]":
    async def _run() -> None:
        rt = settings.runtime
        loop = asyncio.get_running_loop()
        # Sampling period in seconds
        period = max(0.05, min(0.25, rt.saturate_window_ms / 1000.0 / 3.0))
        # Define soft thresholds in seconds
        on_thr = max(0.02, rt.hard_rt_budget_ms / 1000.0 * 2.0)
        off_thr = max(0.01, rt.hard_rt_budget_ms / 1000.0 * 0.7)
        # Exponential moving average of lag
        ema = 0.0
        alpha = 0.25
        # Hysteresis guard
        was_throttled = False
        while True:
            try:
                t0 = loop.time()
                await asyncio.sleep(period)
                now = loop.time()
                lag = max(0.0, now - (t0 + period))
                ema = alpha * lag + (1.0 - alpha) * ema
                if not was_throttled and ema >= on_thr:
                    jx_state.throttle_event.set()
                    was_throttled = True
                elif was_throttled and ema <= off_thr:
                    jx_state.throttle_event.clear()
                    was_throttled = False
            except asyncio.CancelledError:
                raise
            except Exception:
                # Watchdog must never crash; wait and continue
                await asyncio.sleep(0.2)

    return asyncio.create_task(_run(), name="watchdog-service")
