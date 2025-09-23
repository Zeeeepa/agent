"""Logging facade.

Thin re-export of logging helpers from the micro-module implementation.
This keeps the public import path stable while the logic lives under
``jinx.micro.log``.
"""

from __future__ import annotations

from jinx.micro.log.logging import (
    glitch_pulse as glitch_pulse,
    blast_mem as blast_mem,
    bomb_log as bomb_log,
)


__all__ = [
    "glitch_pulse",
    "blast_mem",
    "bomb_log",
]
