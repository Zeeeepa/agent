from __future__ import annotations

"""Conversation orchestrator facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.conversation.orchestrator`` to keep the public API stable.
"""

from typing import Optional
from jinx.micro.conversation.orchestrator import (
    shatter as _shatter,
    corrupt_report as _corrupt_report,
)


async def shatter(x: str, err: Optional[str] = None) -> None:
    return await _shatter(x, err)


async def corrupt_report(err: Optional[str]) -> None:
    return await _corrupt_report(err)
