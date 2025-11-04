from __future__ import annotations

import os
import jinx.state as jx_state


async def dec_pulse(amount: int) -> None:
    """Decrease global pulse.

    Soft mode (default): when pulse falls to <= 0, assert throttle_event to pause
    new conversation processing instead of shutting the runtime down. Hard
    shutdown can be enabled via JINX_PULSE_HARD_SHUTDOWN=true.

    Parameters
    ----------
    amount : int
        Amount to subtract from the current pulse.
    """
    jx_state.pulse -= amount
    if jx_state.pulse <= 0:
        jx_state.pulse = 0
        hard = str(os.getenv("JINX_PULSE_HARD_SHUTDOWN", "0")).strip().lower() in {"1", "true", "yes", "on"}
        
        # Record pulse exhaustion
        try:
            from jinx.micro.runtime.crash_diagnostics import record_operation
            record_operation(
                "pulse_exhausted",
                details={'hard_shutdown': hard, 'amount': amount},
                success=not hard
            )
        except Exception:
            pass
        
        if hard:
            # Signal the runtime to shut down gracefully
            jx_state.shutdown_event.set()
        else:
            # Enter throttled mode; a supervisor task should clear this after cooldown
            jx_state.throttle_event.set()


async def inc_pulse(amount: int) -> None:
    """Increase global pulse by ``amount``.

    Parameters
    ----------
    amount : int
        Amount to add to the current pulse.
    """
    jx_state.pulse += amount
