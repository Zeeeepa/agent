"""Conversation service facade.

Re-exports UI and orchestration bits from micro-modules to keep public API stable.
"""

from __future__ import annotations

from jinx.conversation.ui import pretty_echo
from jinx.conversation.orchestrator import (
    shatter,
    corrupt_report,
    show_sandbox_tail,
)

__all__ = [
    "pretty_echo",
    "shatter",
    "corrupt_report",
    "show_sandbox_tail",
]
