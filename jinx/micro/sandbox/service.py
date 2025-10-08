from __future__ import annotations

from typing import Callable, Awaitable

from jinx.sandbox.executor import blast_zone
from jinx.sandbox.async_runner import run_sandbox


__all__ = ["blast_zone", "arcane_sandbox"]


async def arcane_sandbox(c: str, call: Callable[[str | None], Awaitable[None]] | None = None) -> None:
    """Run code in a separate process and surface results asynchronously."""
    await run_sandbox(c, call)
