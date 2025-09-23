from __future__ import annotations

"""Conversation runner facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.conversation.runner`` to keep the public API stable.
"""

from jinx.micro.conversation.runner import run_blocks as run_blocks


__all__ = [
    "run_blocks",
]
