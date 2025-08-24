from __future__ import annotations

import threading
from typing import Awaitable, Callable
import asyncio

from jinx.sandbox.async_runner import run_sandbox


def launch_sandbox_thread(code: str, callback: Callable[[str | None], Awaitable[None]] | None = None) -> None:
    """Launch the sandboxed run in a daemon thread, bridging to asyncio."""

    def _runner() -> None:
        asyncio.run(run_sandbox(code, callback))

    threading.Thread(target=_runner, daemon=True).start()
