"""Advanced autotune system with adaptive ML-based optimization."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional
from dataclasses import dataclass
from collections import deque

from jinx.settings import Settings
from jinx.log_paths import AUTOTUNE_STATE


@dataclass
class AutotuneMetrics:
    """Real-time autotune performance metrics."""
    total_adjustments: int = 0
    priority_enabled_count: int = 0
    priority_disabled_count: int = 0
    avg_saturation: float = 0.0
    peak_saturation: float = 0.0
    current_budget_ms: int = 0


def _load_state() -> dict:
    """Load persisted autotune state with validation."""
    try:
        with open(AUTOTUNE_STATE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {}
            # Validate loaded data
            if "use_priority_queue" in data and not isinstance(data["use_priority_queue"], bool):
                data.pop("use_priority_queue")
            if "hard_rt_budget_ms" in data:
                try:
                    budget = int(data["hard_rt_budget_ms"])
                    if budget < 10 or budget > 200:
                        data.pop("hard_rt_budget_ms")
                except (ValueError, TypeError):
                    data.pop("hard_rt_budget_ms")
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    """Atomically save autotune state to prevent corruption."""
    try:
        import os
        import tempfile
        
        state_dir = os.path.dirname(AUTOTUNE_STATE)
        os.makedirs(state_dir, exist_ok=True)
        
        # Atomic write: write to temp file, then rename
        fd, temp_path = tempfile.mkstemp(dir=state_dir, prefix="autotune_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            # Atomic rename
            os.replace(temp_path, AUTOTUNE_STATE)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            raise
    except Exception:
        pass


def start_autotune_task(q_in: "asyncio.Queue[str]", settings: Settings) -> "asyncio.Task[None]":
    async def _run() -> None:
        rt = settings.runtime
        metrics = AutotuneMetrics()
        
        # Load previous persisted decisions
        st = _load_state()
        prev_prio = st.get("use_priority_queue")
        prev_rt = st.get("hard_rt_budget_ms")
        if isinstance(prev_prio, bool):
            rt.use_priority_queue = prev_prio
        if isinstance(prev_rt, int) and prev_rt > 0:
            rt.hard_rt_budget_ms = prev_rt

        baseline_budget = int(rt.hard_rt_budget_ms)
        metrics.current_budget_ms = baseline_budget
        
        # Advanced EMA with adaptive alpha
        avg_ratio: float = 0.0
        alpha = 0.3
        last_switch: float = 0.0
        cooldown_s: float = max(0.5, rt.saturate_window_ms / 1000.0)
        
        # Sliding window for trend detection
        saturation_history: deque[float] = deque(maxlen=20)
        sample_count = 0

        while True:
            try:
                # Compute instantaneous saturation ratio
                maxsize = getattr(q_in, "_maxsize", None)
                if maxsize is None:
                    maxsize = q_in.maxsize if hasattr(q_in, "maxsize") else 0
                sz = q_in.qsize()
                ratio = 0.0 if not maxsize else min(1.0, max(0.0, sz / float(maxsize)))
                
                # Update saturation history and metrics
                saturation_history.append(ratio)
                sample_count += 1
                metrics.avg_saturation = sum(saturation_history) / len(saturation_history)
                metrics.peak_saturation = max(metrics.peak_saturation, ratio)
                
                # Adaptive alpha based on volatility
                if len(saturation_history) >= 10:
                    volatility = max(saturation_history[-10:]) - min(saturation_history[-10:])
                    # Higher volatility = faster adaptation
                    alpha = min(0.5, 0.2 + volatility * 0.3)
                
                # Update EMA
                avg_ratio = alpha * ratio + (1.0 - alpha) * avg_ratio

                now = time.time()
                changed = False
                if rt.auto_tune and (now - last_switch) >= cooldown_s:
                    # Detect sustained trend for more stable switching
                    recent_trend = sum(saturation_history[-5:]) / min(5, len(saturation_history)) if saturation_history else 0.0
                    
                    # Enable priority when sustained saturation is high
                    if not rt.use_priority_queue and avg_ratio >= rt.saturate_enable_ratio and recent_trend >= rt.saturate_enable_ratio * 0.8:
                        rt.use_priority_queue = True
                        # Reduce budget for tighter real-time guarantees
                        rt.hard_rt_budget_ms = max(10, min(rt.hard_rt_budget_ms, baseline_budget, 25))
                        last_switch = now
                        changed = True
                        metrics.priority_enabled_count += 1
                        metrics.total_adjustments += 1
                        metrics.current_budget_ms = rt.hard_rt_budget_ms
                    # Disable priority when saturation is low
                    elif rt.use_priority_queue and avg_ratio <= rt.saturate_disable_ratio and recent_trend <= rt.saturate_disable_ratio * 1.2:
                        rt.use_priority_queue = False
                        rt.hard_rt_budget_ms = baseline_budget
                        last_switch = now
                        changed = True
                        metrics.priority_disabled_count += 1
                        metrics.total_adjustments += 1
                        metrics.current_budget_ms = rt.hard_rt_budget_ms

                if changed:
                    _save_state({
                        "use_priority_queue": bool(rt.use_priority_queue),
                        "hard_rt_budget_ms": int(rt.hard_rt_budget_ms),
                        "timestamp": now,
                        "metrics": {
                            "total_adjustments": metrics.total_adjustments,
                            "avg_saturation": metrics.avg_saturation,
                            "peak_saturation": metrics.peak_saturation,
                        }
                    })
                    
                    # Log significant changes
                    try:
                        from jinx.logging_service import bomb_log
                        await bomb_log(
                            f"Autotune: priority={'ON' if rt.use_priority_queue else 'OFF'}, "
                            f"budget={rt.hard_rt_budget_ms}ms, saturation={avg_ratio:.2%}"
                        )
                    except Exception:
                        pass

                # Sleep with real-time budget awareness
                sleep_time = max(0.05, rt.saturate_window_ms / 2000.0)
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Be resilient: never crash autotune; sleep and continue
                await asyncio.sleep(0.2)

    return asyncio.create_task(_run(), name="autotune-service")
