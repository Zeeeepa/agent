from __future__ import annotations

import asyncio
import multiprocessing
import os
from datetime import datetime
from typing import Awaitable, Callable

from jinx.sandbox.executor import blast_zone
from jinx.retry import detonate_payload
from jinx.logging_service import bomb_log
from jinx.log_paths import CLOCKWORK_GHOST
from jinx.sandbox.utils import make_run_log_path, index_run


async def run_sandbox(code: str, callback: Callable[[str | None], Awaitable[None]] | None = None) -> None:
    """Run code in a separate process and surface results asynchronously."""
    with multiprocessing.Manager() as m:
        r = m.dict()

        async def async_sandbox_task() -> None:
            try:
                # Prepare per-run log file to stream output while the process runs
                log_path = make_run_log_path()
                r["log_path"] = log_path

                proc = multiprocessing.Process(
                    target=blast_zone, args=(code, {}, r, log_path)
                )
                proc.start()
                # Strict real-time: yield to the loop without delaying
                while proc.is_alive():
                    await asyncio.sleep(0)
                # Ensure exit code is collected
                proc.join(timeout=0)
            except Exception as e:
                raise Exception(f"Payload mutation error: {e}")

        try:
            # No retries/no delay to avoid any extra waiting for sandbox runs
            await detonate_payload(async_sandbox_task, retries=1, delay=0)
            out, err = r.get("output", ""), r.get("error")
            log_path = r.get("log_path")
            if out:
                await bomb_log(out, CLOCKWORK_GHOST)
            if err:
                await bomb_log(err)
            if log_path:
                await bomb_log(f"Sandbox stream log: {log_path}")
                await index_run(log_path, status=("error" if err else "ok"))
            if callback:
                await callback(err)
        except Exception as e:
            await bomb_log(f"System exile: {e}")
            # Best effort: index as error if log path is known
            try:
                lp = r.get("log_path")
                if lp:
                    await index_run(lp, status="error", error=str(e))
            except Exception:
                pass
