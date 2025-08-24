from __future__ import annotations

import asyncio
import multiprocessing
import os
from datetime import datetime
from typing import Awaitable, Callable

from jinx.sandbox.executor import blast_zone
from jinx.retry import detonate_payload
from jinx.logging_service import bomb_log


async def run_sandbox(code: str, callback: Callable[[str | None], Awaitable[None]] | None = None) -> None:
    """Run code in a separate process and surface results asynchronously."""
    with multiprocessing.Manager() as m:
        r = m.dict()

        async def async_sandbox_task() -> None:
            try:
                # Prepare per-run log file to stream output while the process runs
                log_dir = os.path.join("log", "sandbox")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(
                    log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
                )
                r["log_path"] = log_path

                proc = multiprocessing.Process(
                    target=blast_zone, args=(code, {}, r, log_path)
                )
                proc.start()
                # Asynchronous polling instead of blocking join
                while proc.is_alive():
                    await asyncio.sleep(0.01)
                # Ensure exit code is collected
                proc.join(timeout=0)
            except Exception as e:
                raise Exception(f"Payload mutation error: {e}")

        try:
            await detonate_payload(async_sandbox_task)
            out, err = r.get("output", ""), r.get("error")
            log_path = r.get("log_path")
            if out:
                await bomb_log(out, "log/nano_doppelganger.txt")
            if err:
                await bomb_log(err)
            if log_path:
                await bomb_log(f"Sandbox stream log: {log_path}")
            if callback:
                await callback(err)
        except Exception as e:
            await bomb_log(f"System exile: {e}")
