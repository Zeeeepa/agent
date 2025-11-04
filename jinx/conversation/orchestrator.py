from __future__ import annotations

"""Conversation orchestrator facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.conversation.orchestrator`` to keep the public API stable.
"""

from typing import Optional
from jinx.micro.conversation.orchestrator import (
    shatter as _shatter,
    corrupt_report as _corrupt_report,
)


async def shatter(x: str, err: Optional[str] = None) -> None:
    from jinx.micro.logger.debug_logger import debug_log
    await debug_log(f"shatter called with: {x[:80]}", "WRAPPER")
    # Record conversation request
    try:
        from jinx.micro.runtime.crash_diagnostics import start_operation, end_operation
        start_operation(f"conversation: {x[:50]}")
    except Exception:
        pass
    
    try:
        await debug_log("Calling _shatter...", "WRAPPER")
        result = await _shatter(x, err)
        await debug_log("_shatter returned successfully", "WRAPPER")
        
        # Record success
        try:
            end_operation(success=True)
        except Exception:
            pass
        
        return result
    except Exception as e:
        await debug_log(f"_shatter raised exception: {e}", "WRAPPER")
        # Record failure
        try:
            end_operation(success=False, error=str(e))
        except Exception:
            pass
        raise


async def corrupt_report(err: Optional[str]) -> None:
    return await _corrupt_report(err)
