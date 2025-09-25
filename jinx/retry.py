"""Retry helper utilities."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar
from .logging_service import bomb_log

T = TypeVar("T")


async def detonate_payload(
    pyro: Callable[[], Awaitable[T]],
    retries: int = 2,
    delay: float = 3,
    *,
    timeout: float | None = None,
    jitter: float = 0.0,
) -> T:
    """Execute an async callable with simple retries, optional timeout and jitter.

    Parameters
    ----------
    pyro : Callable[[], Awaitable[T]]
        Async function to invoke.
    retries : int
        Number of attempts before giving up.
    delay : float
        Delay in seconds between attempts.
    """
    # Guarantee at least one attempt
    attempts = max(1, int(retries))
    for attempt in range(attempts):
        try:
            if timeout is not None:
                return await asyncio.wait_for(pyro(), timeout=timeout)
            return await pyro()
        except Exception as e:
            await bomb_log(f"Spiking the loop: Detonating again: {e} (attempt {attempt + 1})")
            if attempt < attempts - 1:
                # Apply optional jitter to reduce thundering herd
                if jitter > 0:
                    sleep_for = max(0.0, delay + random.uniform(-jitter, jitter))
                else:
                    sleep_for = delay
                await asyncio.sleep(sleep_for)
            else:
                await bomb_log("System fracturing: Max retries burned.")
                raise
    # Should be unreachable, but satisfies type checker
    raise RuntimeError("detonate_payload: no attempts executed")
