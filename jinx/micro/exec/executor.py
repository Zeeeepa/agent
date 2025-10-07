from __future__ import annotations

from typing import Awaitable, Callable, Iterable, List, Optional
from jinx.formatters import chain_format
from jinx.logging_service import bomb_log
from jinx.log_paths import TRIGGER_ECHOES
from jinx.sandbox_service import arcane_sandbox
from jinx.codeexec import collect_violations, collect_violations_detailed


async def spike_exec(
    code: str,
    taboo: Iterable[str],
    on_error: Callable[[Optional[str]], Awaitable[None]],
) -> None:
    """Format, validate, and execute code, honoring taboo patterns.

    Parameters
    ----------
    code : str
        Source code to execute.
    taboo : Iterable[str]
        Forbidden substrings that trigger sandboxed execution instead.
    on_error : Callable[[Optional[str]], Awaitable[None]]
        Async callback invoked with error text when execution fails or
        constraints are violated. Receives None on success from the sandbox.
    """
    x = chain_format(code)
    # Enforce prompt constraints before execution (via validators)
    violations: List[str] = collect_violations(x)
    if violations:
        msg = "; ".join(violations)
        await bomb_log(f"Constraint violation: {msg}")
        # Log structured details to aid recovery/debugging
        try:
            dets = collect_violations_detailed(x)
            if dets:
                detail_lines = []
                for d in dets:
                    ident = d.get("id", "rule")
                    line = d.get("line", 1)
                    text = d.get("msg", "")
                    cat = d.get("category", "")
                    detail_lines.append(f"- [{cat}] {ident}@{line}: {text}")
                await bomb_log("\n".join(["Constraint details:"] + detail_lines))
        except Exception:
            pass
        await on_error(msg)
        return
    await bomb_log(x, TRIGGER_ECHOES)
    # Always execute in sandbox to prevent UI lag from busy loops
    await arcane_sandbox(x, call=on_error)
    return
