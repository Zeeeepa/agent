"""Advanced async sandbox runner with process isolation and timeout management."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import time
from datetime import datetime
from typing import Awaitable, Callable, Optional
from dataclasses import dataclass

from jinx.sandbox.executor import blast_zone
from jinx.retry import detonate_payload
from jinx.logging_service import bomb_log
from jinx.log_paths import CLOCKWORK_GHOST
from jinx.sandbox.utils import make_run_log_path, async_rename_run_log
from jinx.micro.exec.run_exports import write_last_run


@dataclass
class SandboxMetrics:
    """Track sandbox execution metrics."""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    timeout_runs: int = 0
    avg_runtime_ms: float = 0.0


_sandbox_metrics = SandboxMetrics()


async def run_sandbox(code: str, callback: Callable[[str | None], Awaitable[None]] | None = None) -> None:
    """Run code in isolated process with advanced timeout and error handling.
    
    Features:
    - Process isolation for safety
    - Hard real-time timeout enforcement
    - Async-friendly design
    - Metrics tracking
    - Stream logging
    - Error detection and recovery
    
    Args:
        code: Python code to execute
        callback: Optional async callback with error message (None on success)
    """
    global _sandbox_metrics
    
    t0 = time.perf_counter()
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
                # Hard RT timeout for sandbox run
                try:
                    max_ms = int(os.getenv("JINX_SANDBOX_MAX_MS", "20000"))
                except Exception:
                    max_ms = 20000
                timeout_s = max(0.1, max_ms / 1000.0)
                # Wait with timeout in a thread to avoid blocking the loop
                await asyncio.to_thread(proc.join, timeout_s)
                if proc.is_alive():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    # Final wait to reap process
                    await asyncio.to_thread(proc.join, 2.0)
                    r["error"] = f"Timeout after {max_ms} ms"
                    r.setdefault("output", "")
            except Exception as e:
                raise Exception(f"Payload mutation error: {e}")

        try:
            try:
                from jinx.observability.otel import span as _span
            except Exception:
                from contextlib import nullcontext as _span  # type: ignore
            # Track execution
            _sandbox_metrics.total_runs += 1
            
            # No retries/no delay to avoid any extra waiting for sandbox runs
            with _span("sandbox.run"):
                await detonate_payload(async_sandbox_task, retries=1, delay=0)
            
            # Update timing metrics
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            count = _sandbox_metrics.total_runs
            _sandbox_metrics.avg_runtime_ms = (
                (_sandbox_metrics.avg_runtime_ms * (count - 1) + elapsed_ms) / count
            )
            
            out, err = r.get("output", ""), r.get("error")
            log_path = r.get("log_path")
            
            if out:
                await bomb_log(out, CLOCKWORK_GHOST)
            # Sentinel-based error detection: if the code explicitly printed an ERROR line,
            # surface it as an error to drive the recovery loop, even without exceptions.
            if not err and out:
                try:
                    for raw in out.splitlines():
                        line = raw.strip()
                        if not line:
                            continue
                        low = line.lower()
                        if low.startswith("error:") or line == "<<JINX_ERROR>>":
                            err = line
                            break
                except Exception:
                    pass
            if err:
                await bomb_log(err)
                # Check if timeout
                if "Timeout" in str(err):
                    _sandbox_metrics.timeout_runs += 1
                _sandbox_metrics.failed_runs += 1
                try:
                    with _span("sandbox.result.error"):
                        pass
                except Exception:
                    pass
            else:
                _sandbox_metrics.successful_runs += 1
                try:
                    with _span("sandbox.result.ok"):
                        pass
                except Exception:
                    pass
            
            if log_path:
                # Rename log file to Jinx-styled status name before announcing path (non-blocking)
                log_path = await async_rename_run_log(log_path, status=("error" if err else "ok"))
                await bomb_log(f"Sandbox stream log: {log_path}")
            # Persist exports for prompt macros (best-effort)
            try:
                write_last_run(out, err, log_path, ok=(err is None))
            except Exception:
                pass
            if callback:
                await callback(err)
        except Exception as e:
            _sandbox_metrics.failed_runs += 1
            await bomb_log(f"System exile: {e}")
            # Best effort: index as error if log path is known
            try:
                lp = r.get("log_path")
                if lp:
                    # Ensure error-styled rename even if exception bubbled up (non-blocking)
                    _ = await async_rename_run_log(lp, status="error")
            except Exception:
                pass


def get_sandbox_metrics() -> SandboxMetrics:
    """Get sandbox execution metrics."""
    return SandboxMetrics(
        total_runs=_sandbox_metrics.total_runs,
        successful_runs=_sandbox_metrics.successful_runs,
        failed_runs=_sandbox_metrics.failed_runs,
        timeout_runs=_sandbox_metrics.timeout_runs,
        avg_runtime_ms=_sandbox_metrics.avg_runtime_ms
    )
