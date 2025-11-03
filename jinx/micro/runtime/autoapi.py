from __future__ import annotations

from typing import Callable, List

# Global registries for auto-start/stop callbacks (module-scope, opt-in)
AUTO_START_FUNCS: List[Callable[..., object]] = []
AUTO_STOP_FUNCS: List[Callable[..., object]] = []


def autostart(fn: Callable[..., object]) -> Callable[..., object]:
    """Decorator to register a function as an auto-start hook.

    The function may accept 0 args or a single PluginContext argument.
    """
    AUTO_START_FUNCS.append(fn)
    return fn


def autostop(fn: Callable[..., object]) -> Callable[..., object]:
    """Decorator to register a function as an auto-stop hook.

    The function may accept 0 args or a single PluginContext argument.
    """
    AUTO_STOP_FUNCS.append(fn)
    return fn


__all__ = [
    "AUTO_START_FUNCS",
    "AUTO_STOP_FUNCS",
    "autostart",
    "autostop",
]
