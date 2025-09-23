from __future__ import annotations

"""Memory parse facade.

Delegates to the micro-module implementation under
``jinx.micro.memory.parse`` while keeping the public API stable.
"""

from jinx.micro.memory.parse import (
    parse_output as parse_output,
    extract as extract,
)


__all__ = [
    "parse_output",
    "extract",
]
