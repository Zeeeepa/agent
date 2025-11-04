from __future__ import annotations

import asyncio
import random
import contextlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set
from enum import Enum

from jinx.settings import Settings


class JobHealth(Enum):
    """Health status of supervised job."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    DEAD = "dead"


@dataclass
class JobMetrics:
    """Advanced metrics for job monitoring."""
    start_count: int = 0
    failure_count: int = 0
    success_count: int = 0
    last_start_time: float = 0.0
    last_failure_time: float = 0.0
    total_runtime_s: float = 0.0
    health: JobHealth = JobHealth.HEALTHY
    consecutive_failures: int = 0


@dataclass(slots=True)
class SupervisedJob:
    name: str
    start: Callable[[], "asyncio.Task[None]"]
    critical: bool = False  # If True, shutdown system if job fails permanently
    health_check: Optional[Callable[[], bool]] = None  # Optional health check callback
    max_restart_rate: float = 5.0  # Max restarts per minute before marking as failing


async def _sleep_cancelable(delay: float, cancel_event: asyncio.Event) -> None:
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=delay)
    except asyncio.TimeoutError:
        return


async def run_supervisor(jobs: list[SupervisedJob], shutdown_event: asyncio.Event, settings: Settings) -> None:
    """
    Advanced supervisor with health monitoring and circuit breaking.

    Features:
    - Auto-restart with exponential backoff
    - Health monitoring and degradation detection
    - Circuit breaking for failing jobs
    - Critical job failure propagation
    - Real-time metrics and alerting
    - Rate-limiting for restart storms
    """
    rt = settings.runtime
    tasks: Dict[str, asyncio.Task[None]] = {}
    restarts: Dict[str, int] = {}
    metrics: Dict[str, JobMetrics] = {j.name: JobMetrics() for j in jobs}
    restart_times: Dict[str, list[float]] = {j.name: [] for j in jobs}

    def _start(name: str) -> None:
        try:
            job_spec = next((j for j in jobs if j.name == name), None)
            if not job_spec:
                return
            t = job_spec.start()
            metrics[name].start_count += 1
            metrics[name].last_start_time = time.time()
        except StopIteration:
            return
        except Exception as e:
            metrics[name].failure_count += 1
            try:
                from jinx.logging_service import bomb_log
                import asyncio as _aio
                _aio.create_task(bomb_log(f"Failed to start job {name}: {e}"))
            except Exception:
                pass
            return
        tasks[name] = t

    # bootstrap all
    for j in jobs:
        _start(j.name)

    try:
        while not shutdown_event.is_set():
            if not tasks:
                # nothing to supervise; await shutdown
                await shutdown_event.wait()
                break
            waiters = set(tasks.values())
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            try:
                done, _ = await asyncio.wait(waiters | {shutdown_task}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not shutdown_task.done():
                    shutdown_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await shutdown_task
            if shutdown_event.is_set():
                break
            # Handle finished tasks
            for t in list(done):
                # Map task to name
                name: Optional[str] = None
                for k, v in list(tasks.items()):
                    if v is t:
                        name = k
                        break
                if name is None:
                    continue
                # Remove from active
                tasks.pop(name, None)
                # Check outcome
                ex = t.exception()
                if isinstance(ex, asyncio.CancelledError):
                    continue
                if ex is None:
                    # Natural completion; do not restart
                    try:
                        from jinx.micro.logger.debug_logger import debug_log_sync
                        debug_log_sync(f"Task '{name}' completed normally - NOT restarting", "SUPERVISOR")
                        debug_log_sync(f"Remaining active tasks: {len(tasks)}", "SUPERVISOR")
                    except Exception:
                        pass
                    continue
                if not rt.supervise_tasks:
                    continue
                count = restarts.get(name, 0)
                if count >= rt.autorestart_limit:
                    # Give up on this job
                    m = metrics[name]
                    m.health = JobHealth.DEAD
                    
                    # Check if critical job - trigger shutdown
                    job_spec = next((j for j in jobs if j.name == name), None)
                    if job_spec and job_spec.critical:
                        try:
                            from jinx.logging_service import bomb_log
                            await bomb_log(f"CRITICAL JOB DEAD: {name} - initiating shutdown")
                        except Exception:
                            pass
                        shutdown_event.set()
                    continue
                
                # Rate limiting: check restart frequency
                current_time = time.time()
                restart_times[name].append(current_time)
                # Keep only last minute of restart times
                restart_times[name] = [t for t in restart_times[name] if (current_time - t) < 60.0]
                
                restart_rate = len(restart_times[name])
                job_spec = next((j for j in jobs if j.name == name), None)
                if job_spec and restart_rate > job_spec.max_restart_rate:
                    # Too many restarts - mark as failing and slow down
                    m = metrics[name]
                    m.health = JobHealth.FAILING
                    m.consecutive_failures = count
                    try:
                        from jinx.logging_service import bomb_log
                        await bomb_log(f"Job {name} restarting too fast ({restart_rate}/min) - backing off")
                    except Exception:
                        pass
                else:
                    m = metrics[name]
                    if m.consecutive_failures >= 2:
                        m.health = JobHealth.DEGRADED
                    m.consecutive_failures = count
                
                restarts[name] = count + 1
                m.start_count += 1
                m.last_start_time = current_time
                
                # Compute backoff with jitter
                base = max(1, rt.backoff_min_ms) / 1000.0
                cap = max(base, rt.backoff_max_ms / 1000.0)
                delay = min(cap, base * (2 ** count))
                delay = delay * (0.7 + 0.6 * random.random())
                
                # Additional delay for failing jobs
                if m.health == JobHealth.FAILING:
                    delay *= 2.0
                
                # Wait unless shutdown requested
                await _sleep_cancelable(delay, shutdown_event)
                if shutdown_event.is_set():
                    break
                _start(name)
    finally:
        # cancel all active (with recursion protection)
        for t in tasks.values():
            try:
                t.cancel()
            except RecursionError:
                # Skip tasks that cause deep recursion
                pass
        
        # Wait for tasks with timeout to prevent hanging
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks.values(), return_exceptions=True),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                # Some tasks didn't finish, that's acceptable
                pass
            except RecursionError:
                # Deep recursion in gather, bail out
                pass
