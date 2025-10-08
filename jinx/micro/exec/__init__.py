from __future__ import annotations

from typing import Any

__all__ = ["spike_exec"]


def __getattr__(name: str) -> Any:
    if name == "spike_exec":
        from .executor import spike_exec  # local import to avoid circular deps
        return spike_exec
    raise AttributeError(name)
