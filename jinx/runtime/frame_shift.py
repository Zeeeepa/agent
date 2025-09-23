from __future__ import annotations

"""Runtime frame_shift facade.

Delegates to the micro-module implementation under
``jinx.micro.runtime.frame_shift`` while keeping the public API stable.
"""

from jinx.micro.runtime.frame_shift import frame_shift as frame_shift


__all__ = [
    "frame_shift",
]
