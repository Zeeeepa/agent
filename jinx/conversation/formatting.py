from __future__ import annotations

"""Conversation formatting facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.conversation.formatting`` to keep the public API stable.
"""

from jinx.micro.conversation.formatting import (
    ensure_header_block_separation as ensure_header_block_separation,
    build_header as build_header,
)


__all__ = [
    "ensure_header_block_separation",
    "build_header",
]
