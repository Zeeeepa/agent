"""Code execution facade.

Delegates to the micro-module implementation under
``jinx.micro.exec.executor`` while keeping the public API stable.
"""

from __future__ import annotations

from jinx.micro.exec.executor import spike_exec as spike_exec


__all__ = [
    "spike_exec",
]
