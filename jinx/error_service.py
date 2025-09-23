"""Error and pulse management facade.

Delegates to the micro-module implementation under
``jinx.micro.core.error`` to keep the public API stable.
"""

from __future__ import annotations

from jinx.micro.core.error import (
    dec_pulse as dec_pulse,
    inc_pulse as inc_pulse,
)


__all__ = [
    "dec_pulse",
    "inc_pulse",
]
