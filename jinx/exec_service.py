"""Code execution service.

Formats and executes model-generated code within a constrained context,
enforcing prompt-specified restrictions and reporting output and errors.

This module intentionally keeps execution semantics simple: format the code,
perform fast string-based constraint checks, and either execute inline or
delegate to a sandbox when taboo patterns are detected. It is not a secure
isolation mechanism.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Iterable, List
from jinx.formatters import chain_format
from .logging_service import bomb_log
from jinx.log_paths import TRIGGER_ECHOES
from .sandbox_service import arcane_sandbox
from jinx.codeexec import collect_violations


async def spike_exec(
    code: str,
    taboo: Iterable[str],
    on_error: Callable[[str], Awaitable[None]],
) -> None:
    """Format, validate, and execute code, honoring taboo patterns.

    Parameters
    ----------
    code : str
        Source code to execute.
    taboo : Iterable[str]
        Forbidden substrings that trigger sandboxed execution instead.
    on_error : Callable[[str], Awaitable[None]]
        Async callback invoked with error text when execution fails or
        constraints are violated.
    """
    x = chain_format(code)
    # Enforce prompt constraints before execution (via validators)
    violations: List[str] = collect_violations(x)
    if violations:
        msg = "; ".join(violations)
        await bomb_log(f"Constraint violation: {msg}")
        await on_error(msg)
        return
    await bomb_log(x, TRIGGER_ECHOES)
    # Always execute in sandbox to prevent UI lag from busy loops
    await arcane_sandbox(x, call=on_error)
    return
