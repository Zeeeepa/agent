from __future__ import annotations

"""Runtime input_task facade.

Delegates to the micro-module implementation under
``jinx.micro.runtime.input_task`` while keeping the public API stable.
"""

from jinx.micro.runtime.input_task import start_input_task as start_input_task


__all__ = [
    "start_input_task",
]
