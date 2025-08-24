"""Sandboxed execution helpers.

Fully async facade over the sandbox runner. Avoids threads and blocking calls.
Executes untrusted snippets in an isolated process, capturing stdout/stderr and
posting results to logs. Non-blocking via a daemon thread.
"""

from __future__ import annotations

from typing import Callable, Awaitable
from jinx.sandbox import blast_zone
from jinx.sandbox.async_runner import run_sandbox


__all__ = ["blast_zone", "arcane_sandbox"]


async def arcane_sandbox(c: str, call: Callable[[str | None], Awaitable[None]] | None = None) -> None:
    """Run code in a separate process and surface results asynchronously.

    Parameters
    ----------
    c : str
        Code to execute within sandbox.
    call : Optional[Callable[[str | None], Awaitable[None]]]
        Optional async callback receiving an error string or None when finished.
    """
    await run_sandbox(c, call)
