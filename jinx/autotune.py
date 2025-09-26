from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from jinx.settings import Settings
from jinx.log_paths import AUTOTUNE_STATE


def _load_state() -> dict:
    try:
        with open(AUTOTUNE_STATE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    try:
        import os
        os.makedirs(os.path.dirname(AUTOTUNE_STATE), exist_ok=True)
        with open(AUTOTUNE_STATE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def start_autotune_task(q_in: "asyncio.Queue[str]", settings: Settings) -> "asyncio.Task[None]":
    async def _run() -> None:
        rt = settings.runtime
        # Load previous persisted decisions
        st = _load_state()
        prev_prio = st.get("use_priority_queue")
        prev_rt = st.get("hard_rt_budget_ms")
        if isinstance(prev_prio, bool):
            rt.use_priority_queue = prev_prio
        if isinstance(prev_rt, int) and prev_rt > 0:
            rt.hard_rt_budget_ms = prev_rt

        baseline_budget = int(rt.hard_rt_budget_ms)
        avg_ratio: float = 0.0  # EMA of saturation
        alpha = 0.3
        last_switch: float = 0.0
        cooldown_s: float = max(0.5, rt.saturate_window_ms / 1000.0)

        while True:
            try:
                # Compute instantaneous saturation ratio
                maxsize = getattr(q_in, "_maxsize", None)
                if maxsize is None:
                    maxsize = q_in.maxsize if hasattr(q_in, "maxsize") else 0
                sz = q_in.qsize()
                ratio = 0.0 if not maxsize else min(1.0, max(0.0, sz / float(maxsize)))
                # Update EMA
                avg_ratio = alpha * ratio + (1.0 - alpha) * avg_ratio

                now = time.time()
                changed = False
                if rt.auto_tune and (now - last_switch) >= cooldown_s:
                    # Enable priority when sustained saturation is high
                    if not rt.use_priority_queue and avg_ratio >= rt.saturate_enable_ratio:
                        rt.use_priority_queue = True
                        rt.hard_rt_budget_ms = max(10, min(rt.hard_rt_budget_ms, baseline_budget, 25))
                        last_switch = now
                        changed = True
                    # Disable priority when saturation is low
                    elif rt.use_priority_queue and avg_ratio <= rt.saturate_disable_ratio:
                        rt.use_priority_queue = False
                        rt.hard_rt_budget_ms = baseline_budget
                        last_switch = now
                        changed = True

                if changed:
                    _save_state({
                        "use_priority_queue": bool(rt.use_priority_queue),
                        "hard_rt_budget_ms": int(rt.hard_rt_budget_ms),
                    })

                # Sleep a short while to avoid busy-loop; granularity driven by window
                await asyncio.sleep(max(0.05, rt.saturate_window_ms / 2000.0))
            except asyncio.CancelledError:
                raise
            except Exception:
                # Be resilient: never crash autotune; sleep and continue
                await asyncio.sleep(0.2)

    return asyncio.create_task(_run(), name="autotune-service")
