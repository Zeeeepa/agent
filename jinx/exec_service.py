"""Code execution service.

Formats and executes model-generated code within a constrained context,
enforcing prompt-specified restrictions and reporting output and errors.

This module intentionally keeps execution semantics simple: format the code,
perform fast string-based constraint checks, and either execute inline or
delegate to a sandbox when taboo patterns are detected. It is not a secure
isolation mechanism.
"""

from __future__ import annotations

import traceback
from typing import Awaitable, Callable, Iterable, List
from .format_service import warp_blk
from .logging_service import bomb_log
from .sandbox_service import arcane_sandbox
from jinx.codeexec import collect_violations
from jinx.codeexec.runner.inline import run_inline


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
    x = warp_blk(code)
    # Enforce prompt constraints before execution (via validators)
    violations: List[str] = collect_violations(x)
    if violations:
        msg = "; ".join(violations)
        await bomb_log(f"Constraint violation: {msg}")
        await on_error(msg)
        return
    await bomb_log(x, "log/detonator.txt")
    if any((z in x for z in taboo)):
        arcane_sandbox(x, call=on_error)
        return
    try:
        out = run_inline(x)
        if out:
            await bomb_log(out, "log/nano_doppelganger.txt")
            # Echo to terminal as well
            import sys as _sys
            _sys.stdout.write(out)
            _sys.stdout.flush()
    except Exception:
        err = traceback.format_exc()
        await bomb_log(err)
        await on_error(err)
