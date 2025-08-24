"""Spinner service.

Renders a non-blocking terminal spinner while background tasks are running.
The spinner terminates when the provided event is set.
"""

from __future__ import annotations

import asyncio, time, sys
from jinx.bootstrap import ensure_optional
import importlib
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

    def _can_render(s: str) -> bool:
        try:
            s.encode(enc)
            return True
        except Exception:
            return False

    unicode_ok = _can_render("⠋") and _can_render("\u2764")

    # Use a smoother, multi-frame spinner when Unicode is available
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if unicode_ok else "-\\|/"
    heart = "\u2764" if unicode_ok else "<3"

    fx = print_formatted_text
    ft = FormattedText
    t0 = time.perf_counter()

    while not evt.is_set():
        dt = time.perf_counter() - t0
        # Slightly higher frame rate with more frames for smoothness
        idx = int(dt * 14) % len(spinner)
        zz = spinner[idx]

        # Dots cadence
        n = int(dt * 0.8) % 4
        dd = "." * n

        # Display
        hf = heart
        pulse = state.pulse
        clr = "ansibrightgreen"
        fx(ft([(clr, f"{hf} {pulse} {dd} {zz} Processing {dt:.3f}s")]), end="\r", flush=True)
        await asyncio.sleep(0.035)

    fx(ft([("", " " * 80)]), end="\r", flush=True)
