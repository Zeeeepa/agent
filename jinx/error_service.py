"""Error and pulse management service.

This module manages the global "pulse" value used by the spinner and provides
helpers to increase/decrease it. When pulse depletes to zero or below, the
process exits with a non-zero code to signal failure. This keeps failure
propagation simple for CLI usage.
"""

from __future__ import annotations

import sys
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
        sys.exit(1)


async def inc_pulse(amount: int) -> None:
    """Increase global pulse by ``amount``.

    Parameters
    ----------
    amount : int
        Amount to add to the current pulse.
    """
    jx_state.pulse += amount
