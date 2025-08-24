from __future__ import annotations

import asyncio
import multiprocessing
from typing import Awaitable, Callable

from jinx.sandbox.executor import blast_zone
from jinx.retry import detonate_payload
from jinx.logging_service import bomb_log


async def run_sandbox(code: str, callback: Callable[[str | None], Awaitable[None]] | None = None) -> None:
    """Run code in a separate process and surface results asynchronously."""
    with multiprocessing.Manager() as m:
        r = m.dict()

        def sandbox_task() -> None:
            try:
                proc = multiprocessing.Process(target=blast_zone, args=(code, {}, r))
                proc.start()
                proc.join()
            except Exception as e:
                raise Exception(f"Payload mutation error: {e}")

        async def async_sandbox_task() -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sandbox_task)

        try:
            await detonate_payload(async_sandbox_task)
            out, err = r.get("output", ""), r.get("error")
            if out:
                await bomb_log(out, "log/nano_doppelganger.txt")
            if err:
                await bomb_log(err)
            if callback:
                await callback(err)
        except Exception as e:
            await bomb_log(f"System exile: {e}")
