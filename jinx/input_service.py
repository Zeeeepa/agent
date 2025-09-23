"""Input facade.

Delegates interactive input loop to the micro-module implementation under
``jinx.micro.io.input`` while keeping the public API stable.
"""

from __future__ import annotations

from jinx.micro.io.input import neon_input as neon_input


__all__ = [
    "neon_input",
]
