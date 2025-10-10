from __future__ import annotations

from jinx.sandbox.utils import read_latest_sandbox_tail
from jinx.micro.ui.output import pretty_echo


async def show_sandbox_tail() -> None:
    """Print the latest sandbox log (full if short, else last N lines)."""
    content, _ = read_latest_sandbox_tail()
    if content is not None:
        pretty_echo(content, title="Sandbox")
