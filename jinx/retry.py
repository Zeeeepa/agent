"""Retry helper utilities."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar
from .logging_service import bomb_log

T = TypeVar("T")


async def detonate_payload(pyro: Callable[[], Awaitable[T]], retries: int = 2, delay: float = 3) -> T:
    """Execute an async callable with simple retries and delay.

    Parameters
    ----------
    pyro : Callable[[], Awaitable[T]]
        Async function to invoke.
    retries : int
        Number of attempts before giving up.
    delay : float
        Delay in seconds between attempts.
    """
    for attempt in range(retries):
        try:
            return await pyro()
        except Exception as e:
            await bomb_log(f"Spiking the loop: Detonating again: {e} (attempt {attempt + 1})")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                await bomb_log("System fracturing: Max retries burned.")
                raise
