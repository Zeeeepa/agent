from __future__ import annotations

"""Memory storage facade.

Delegates to the micro-module implementation under
``jinx.micro.memory.storage`` while keeping the public API stable.
"""

from jinx.micro.memory.storage import (
    read_evergreen as read_evergreen,
    write_state as write_state,
    ensure_nl as ensure_nl,
)


__all__ = [
    "read_evergreen",
    "write_state",
    "ensure_nl",
]
