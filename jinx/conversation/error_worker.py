from __future__ import annotations

"""Error worker facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.conversation.error_worker`` to keep the public API stable.
"""

from jinx.micro.conversation.error_worker import (
    enqueue_error_retry as enqueue_error_retry,
    stop_error_worker as stop_error_worker,
)


__all__ = [
    "enqueue_error_retry",
    "stop_error_worker",
]
