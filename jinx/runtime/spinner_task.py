from __future__ import annotations

import asyncio
from jinx.spinner_service import sigil_spin


def start_spinner(evt: asyncio.Event) -> asyncio.Task[None]:
    """Start a spinner task that stops when evt is set."""
    return asyncio.create_task(sigil_spin(evt))
