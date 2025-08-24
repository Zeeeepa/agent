"""Spinner service.

Renders a non-blocking terminal spinner while background tasks are running.
The spinner terminates when the provided event is set.
"""

from __future__ import annotations

import asyncio, time, sys, random
from jinx.bootstrap import ensure_optional
import importlib
from jinx.spinner.phrases import PHRASES as phrases
from jinx.spinner import ascii_mode as _ascii_mode, can_render as _can_render
from jinx.spinner import get_spinner_frames, get_hearts
import jinx.state as state

# Lazy import with auto-install of prompt_toolkit
ensure_optional(["prompt_toolkit"])  # installs if missing
print_formatted_text = importlib.import_module("prompt_toolkit").print_formatted_text  # type: ignore[assignment]
FormattedText = importlib.import_module("prompt_toolkit.formatted_text").FormattedText  # type: ignore[assignment]


async def sigil_spin(evt: asyncio.Event) -> None:
    """Minimal, pretty spinner that shows pulse and spins until evt is set.

    Parameters
    ----------
    evt : asyncio.Event
        Event signaling spinner shutdown.
    """
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

    while not evt.is_set():
        dt = time.perf_counter() - t0
        pulse = state.pulse
        clr = "ansibrightgreen"

        # Change phrase a bit slower (~every 0.85s)
        if (dt - last_change) >= 0.85:
            phrase_idx = random.randrange(len(phrases))
            last_change = dt
        phrase = phrases[phrase_idx]

        # Loading dots cadence (0..3 dots cycling)
        n = int(dt * 0.8) % 4
        dd = "." * n

        # Pulsating heart (toggle ~1.5 Hz) with minimal size change
        beat = int(dt * 1.5) % 2
        heart = heart_a if beat == 0 else heart_b
        style = clr if beat == 0 else f"{clr} bold"

        # ASCII spinner right after pulse (~12 FPS)
        sidx = int(dt * 12) % len(spin_frames)
        spin = spin_frames[sidx]

        fx(ft([(style, f"{heart} {pulse} {spin} {dd} {phrase} {dt:.3f}s")]), end="\r", flush=True)
        await asyncio.sleep(0.035)

    fx(ft([("", " " * 80)]), end="\r", flush=True)
