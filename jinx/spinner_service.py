"""Spinner facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.ui.spinner`` to keep the public API stable.
"""

from __future__ import annotations

from jinx.micro.ui.spinner import sigil_spin as sigil_spin


__all__ = [
    "sigil_spin",
]
