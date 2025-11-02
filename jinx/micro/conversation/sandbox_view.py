from __future__ import annotations

from jinx.sandbox.utils import read_latest_sandbox_tail
from jinx.micro.ui.output import pretty_echo_async as _pretty_echo_async, pretty_echo


async def show_sandbox_tail() -> None:
    """Print the latest sandbox log (full if short, else last N lines)."""
    content, _ = read_latest_sandbox_tail()
    if content is not None:
        try:
            await _pretty_echo_async(content, title="Sandbox")
        except Exception:
            # Fallback to sync printing in a thread
            import asyncio as _aio
            try:
                await _aio.to_thread(pretty_echo, content, "Sandbox")
            except Exception:
                pass
