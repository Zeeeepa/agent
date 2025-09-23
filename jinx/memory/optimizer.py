from __future__ import annotations

"""Memory optimizer facade.

Delegates to the micro-module implementation under
``jinx.micro.memory.optimizer`` while keeping the public API stable.
"""

from jinx.micro.memory.optimizer import (
    submit as submit,
    stop as stop,
    start_memory_optimizer_task as start_memory_optimizer_task,
)


__all__ = [
    "submit",
    "stop",
    "start_memory_optimizer_task",
]
