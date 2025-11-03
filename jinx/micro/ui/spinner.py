from __future__ import annotations

"""Spinner micro-module.

Renders a non-blocking terminal spinner while background tasks are running.
The spinner terminates when the provided event is set.
"""

import asyncio
import time
import sys
import random
import importlib
import os

from jinx.bootstrap import ensure_optional
from jinx.spinner.phrases import PHRASES as phrases
from jinx.spinner import ascii_mode as _ascii_mode, can_render as _can_render
from jinx.spinner import get_spinner_frames, get_hearts
import jinx.state as state
from .spinner_util import format_activity_detail, parse_env_bool, parse_env_int

# Lazy import with auto-install of prompt_toolkit
ensure_optional(["prompt_toolkit"])  # installs if missing
print_formatted_text = importlib.import_module("prompt_toolkit").print_formatted_text  # type: ignore[assignment]
FormattedText = importlib.import_module("prompt_toolkit.formatted_text").FormattedText  # type: ignore[assignment]
patch_stdout = importlib.import_module("prompt_toolkit.patch_stdout").patch_stdout  # type: ignore[assignment]


async def sigil_spin(evt: asyncio.Event) -> None:
    """Minimal, pretty spinner that shows pulse and spins until evt is set.

    Parameters
    ----------
    evt : asyncio.Event
        Event signaling spinner shutdown.
    """
    mode = (os.getenv("JINX_SPINNER_MODE", "toolbar").strip().lower() or "toolbar")
    if mode == "toolbar":
        # Toolbar mode: do not print; let PromptSession bottom_toolbar render from state
        try:
            state.spin_on = True
            state.spin_t0 = float(time.perf_counter())
            last_tick = time.perf_counter()
            lag_ema_ms = 0.0
            alpha = 0.3
            hi_cnt = 0
            lo_cnt = 0
            while not evt.is_set():
                now = time.perf_counter()
                gap = now - last_tick
                last_tick = now
                # Nominal interval ~0.08s
                over_ms = max(0.0, (gap - 0.08) * 1000.0)
                lag_ema_ms = alpha * over_ms + (1.0 - alpha) * lag_ema_ms
                # Expose lag to state for diagnostics/autotune
                try:
                    state.lag_ema_ms = float(lag_ema_ms)
                except Exception:
                    pass
                # Simple hysteresis: raise throttle on sustained high lag, clear on sustained low lag
                try:
                    if lag_ema_ms > 120.0:
                        hi_cnt += 1; lo_cnt = 0
                    elif lag_ema_ms < 50.0:
                        lo_cnt += 1; hi_cnt = 0
                    else:
                        hi_cnt = max(0, hi_cnt - 1); lo_cnt = max(0, lo_cnt - 1)
                    # Clear via TTL first if present
                    try:
                        tz = float(getattr(state, "throttle_unset_ts", 0.0) or 0.0)
                    except Exception:
                        tz = 0.0
                    if tz and now >= tz:
                        state.throttle_event.clear()
                        state.throttle_unset_ts = 0.0
                    elif hi_cnt >= 4:
                        state.throttle_event.set()
                    elif lo_cnt >= 8 and not tz:
                        state.throttle_event.clear()
                except Exception:
                    pass
                await asyncio.sleep(0.08)
        finally:
            state.spin_on = False
        return

    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    fx = print_formatted_text
    ft = FormattedText
    t0 = time.perf_counter()
    ascii_mode = _ascii_mode()
    heart_a, heart_b = get_hearts(ascii_mode, can=lambda s: _can_render(s, enc))
    phrase_idx = 0
    last_change = 0.0  # seconds since t0 when phrase last changed
    # Thin circular spinner frames (Unicode) with ASCII fallback
    spin_frames = get_spinner_frames(ascii_mode, can=lambda s: _can_render(s, enc))
    # Event loop lag EMA tracking
    last_tick = time.perf_counter()
    lag_ema_ms = 0.0
    alpha = 0.3
    # Redraw throttling and change detection
    last_emit = t0
    last_stage = None
    last_tasks = None
    last_line = ""

    # Ensure spinner writes cooperate with the active prompt by patching stdout
    try:
        state.spin_on = True
        state.spin_t0 = float(time.perf_counter())
        with patch_stdout(raw=True):
            while not evt.is_set():
                now = time.perf_counter()
                dt = now - t0
                pulse = state.pulse
                clr = "ansibrightgreen"
                # Change phrase a bit slower (~every 0.85s)
                if (dt - last_change) >= 0.85:
                    new_idx = random.randrange(len(phrases)) if phrases else 0
                    if phrases and len(phrases) > 1 and new_idx == phrase_idx:
                        new_idx = (new_idx + 1) % len(phrases)
                    phrase_idx = new_idx
                    last_change = dt
                phrase = phrases[phrase_idx]

                # Loading dots cadence (0..3 dots cycling)
                n = int(dt * 0.8) % 4
                dd = "." * n

                # Pulsating heart (toggle ~1.5 Hz) with minimal size change
                beat = int(dt * 1.5) % 2
                heart = heart_a if beat == 0 else heart_b
                style = clr if beat == 0 else f"{clr} bold"

                # ASCII spinner right after pulse (~10–12 FPS)
                sidx = int(dt * 10) % len(spin_frames)
                spin = spin_frames[sidx]

                # Compose dynamic activity description
                show_act = parse_env_bool("JINX_SPINNER_ACTIVITY", True)
                desc = ""
                if show_act:
                    try:
                        act = (getattr(state, "activity", "") or "").strip()
                        if act:
                            age = 0.0
                            try:
                                age = max(0.0, time.perf_counter() - float(getattr(state, "activity_ts", 0.0) or 0.0))
                            except Exception:
                                age = 0.0
                            desc = f" | {act} [{age:.1f}s]"
                    except Exception:
                        desc = ""

                # Compose compact detail from structured activity_detail if enabled
                show_det = parse_env_bool("JINX_SPINNER_ACTIVITY_DETAIL", True)
                det_str = ""
                stage = None
                tasks = None
                if show_det:
                    try:
                        det = getattr(state, "activity_detail", None)
                        det_str2, stage2, tasks2 = format_activity_detail(det)
                        det_str = det_str2
                        if stage2 is not None:
                            stage = stage2
                        if tasks2 is not None:
                            tasks = tasks2
                    except Exception:
                        det_str = ""

                # Compute event loop lag EMA (approximate): expected ~0.06s per tick
                gap = now - last_tick
                last_tick = now
                show_lag = parse_env_bool("JINX_SPINNER_SHOW_LAG", False)
                if show_lag:
                    # Only accumulate positive overruns over nominal interval
                    over_ms = max(0.0, (gap - 0.06) * 1000.0)
                    lag_ema_ms = alpha * over_ms + (1.0 - alpha) * lag_ema_ms
                # Expose lag EMA and color adapt based on stage/lag (optional)
                try:
                    try:
                        state.lag_ema_ms = float(lag_ema_ms)
                    except Exception:
                        pass
                    if stage and str(stage).startswith("repair"):
                        clr = "ansibrightcyan"
                    elif show_lag and lag_ema_ms > 120.0:
                        clr = "ansired"
                    elif show_lag and lag_ema_ms > 50.0:
                        clr = "ansiyellow"
                    else:
                        clr = "ansibrightgreen"
                except Exception:
                    clr = "ansibrightgreen"
                # Lag-driven throttle hysteresis in print mode as well
                try:
                    # TTL-based clear takes precedence
                    try:
                        tz = float(getattr(state, "throttle_unset_ts", 0.0) or 0.0)
                    except Exception:
                        tz = 0.0
                    if tz and now >= tz:
                        state.throttle_event.clear()
                        state.throttle_unset_ts = 0.0
                    elif lag_ema_ms > 120.0:
                        state.throttle_event.set()
                    elif lag_ema_ms < 50.0 and not tz:
                        state.throttle_event.clear()
                except Exception:
                    pass
                # Build single-line render with optional lag
                lag_str = (f" lag:{lag_ema_ms:.1f}ms" if show_lag else "")
                line = f"{heart} {pulse} {spin} {dd} {phrase}{desc}{det_str}{lag_str} (total {dt:.1f}s)"

                # Redraw throttling: only when content changes significantly or interval elapsed
                min_ms = parse_env_int("JINX_SPINNER_MIN_UPDATE_MS", 160)
                redraw_on_change = parse_env_bool("JINX_SPINNER_REDRAW_ONLY_ON_CHANGE", True)

                significant_change = (stage is not None and stage != last_stage) or (tasks is not None and tasks != last_tasks)
                time_elapsed = (now - last_emit) * 1000.0 >= float(min_ms)
                content_changed = (line != last_line)

                if (not redraw_on_change and time_elapsed) or (redraw_on_change and (significant_change or (time_elapsed and content_changed))):
                    fx(ft([(style, line)]), end="\r", flush=False)
                    last_emit = now
                    last_line = line
                    last_stage = stage if stage is not None else last_stage
                    last_tasks = tasks if tasks is not None else last_tasks

                await asyncio.sleep(0.06)
    finally:
        state.spin_on = False
        
    
    fx(ft([("", " " * 80)]), end="\r", flush=True)


__all__ = [
    "sigil_spin",
]
