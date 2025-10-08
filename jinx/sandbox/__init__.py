from __future__ import annotations

from typing import Any

__all__ = ["blast_zone", "run_sandbox"]


def __getattr__(name: str) -> Any:
    if name == "blast_zone":
        from .executor import blast_zone  # lazy import to avoid cycles
        return blast_zone
    if name == "run_sandbox":
        from .async_runner import run_sandbox  # lazy import to avoid cycles
        return run_sandbox
    raise AttributeError(name)
