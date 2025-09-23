from __future__ import annotations

"""Conversation UI facade.

Delegates to the micro-module implementation under
``jinx.micro.ui.output`` while keeping the public API stable.
"""

from jinx.micro.ui.output import pretty_echo as pretty_echo


__all__ = [
    "pretty_echo",
]
