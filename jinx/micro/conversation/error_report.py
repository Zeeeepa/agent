from __future__ import annotations

from typing import Optional

from jinx.logging_service import bomb_log
from jinx.error_service import dec_pulse
from jinx.micro.conversation.error_worker import enqueue_error_retry


async def corrupt_report(err: Optional[str]) -> None:
    """Log an error, enqueue a serialized retry, and decay pulse."""
    if err is None:
        return
    await bomb_log(err)
    # Enqueue follow-up step to be processed by a single worker
    await enqueue_error_retry(err)
    await dec_pulse(30)
