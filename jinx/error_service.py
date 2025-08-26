"""Error and pulse management service.

This module manages the global "pulse" value used by the spinner and provides
helpers to increase/decrease it. When pulse depletes to zero or below, the
runtime initiates a graceful shutdown by signalling a global event instead of
raising SystemExit from background tasks. This prevents noisy terminal errors
while ensuring a clean exit path.
"""

from __future__ import annotations

import jinx.state as jx_state


async def dec_pulse(amount: int) -> None:
    """Decrease global pulse and exit when depleted.

    Parameters
    ----------
    amount : int
        Amount to subtract from the current pulse.
    """
    jx_state.pulse -= amount
    if jx_state.pulse <= 0:
        # Signal the runtime to shut down gracefully
        jx_state.shutdown_event.set()


async def inc_pulse(amount: int) -> None:
    """Increase global pulse by ``amount``.

    Parameters
    ----------
    amount : int
        Amount to add to the current pulse.
    """
    jx_state.pulse += amount
