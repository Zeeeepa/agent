from __future__ import annotations

import asyncio
import random
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set
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
    # Per-job recovery state for staged escalation and cooldowns
    recovery_state: Dict[str, Dict[str, Any]] = {}

    def _cooldown_ok(job: str, key: str, cd_s: float) -> bool:
        st = recovery_state.setdefault(job, {})
        now = time.time()
        last = float(st.get(f"ts:{key}") or 0.0)
        if (now - last) >= max(0.1, cd_s):
            st[f"ts:{key}"] = now
            return True
        return False

    async def _auto_recover(task_name: str, ex: BaseException) -> None:
        """Best-effort auto-recovery pipeline for failed supervised jobs.

        Steps (all optional/no-op on error):
        - Log event
        - Invoke self-healing with exception details
        - Schedule resilience repairs for missing modules
        - If Import/ModuleNotFound, request self-update apply via task API
        """
        # Prepare traceback string
        try:
            import traceback as _tb
            tb = "".join(_tb.format_exception(type(ex), ex, ex.__traceback__))
        except Exception:
            tb = ""
        try:
            from jinx.observability.otel import span as _span
        except Exception:
            from contextlib import nullcontext as _span  # type: ignore
        with _span("supervisor.auto_recover", attrs={"task": task_name, "exc": type(ex).__name__}):
            # Publish helper (best-effort)
            try:
                from jinx.micro.runtime.plugins import publish_event as _pub_evt
            except Exception:
                def _pub_evt(_name: str, _payload: Dict[str, Any]) -> None:  # type: ignore
                    return
            # Update per-job failure counters/state
            st = recovery_state.setdefault(task_name, {})
            try:
                st["fails"] = int(st.get("fails", 0)) + 1
            except Exception:
                st["fails"] = 1
            fails_n = int(st.get("fails", 1))
            # Emit base event
            try:
                _pub_evt("recovery.event", {"task": task_name, "type": type(ex).__name__, "fails": fails_n})
            except Exception:
                pass
            # Log
            try:
                from jinx.logging_service import bomb_log
                await bomb_log(f"Supervisor auto-recover for {task_name}: {type(ex).__name__}: {ex}")
            except Exception:
                pass
            # Quality scan (early stage) with cooldown to surface issues quickly
            if _cooldown_ok(task_name, "quality_scan", 15.0):
                try:
                    _pub_evt("recovery.stage", {"task": task_name, "stage": "quality_scan"})
                except Exception:
                    pass
                try:
                    import re as _re, os as _os
                    files: list[str] = []
                    cwd = (_os.getcwd() or "").strip()
                    if tb:
                        for mm in _re.finditer(r'File "([^"]+\.py)", line (\d+)', tb):
                            ap = str(mm.group(1) or "").strip()
                            ln = int(mm.group(2) or 0)
                            if ap and cwd and ap.lower().startswith(cwd.lower()):
                                rel = ap[len(cwd):].lstrip("/\\")
                                files.append(f"{rel}:{ln}")
                                break
                    # Fallback: recently modified tracked by reprogrammer (if any)
                    if not files:
                        try:
                            import jinx.state as jx_state
                            mods = list(getattr(jx_state, "reprogram_modified", []) or [])
                            for ap in mods[:4]:
                                rel = ap[len(cwd):].lstrip("/\\") if ap.lower().startswith(cwd.lower()) else ap
                                files.append(rel)
                        except Exception:
                            pass
                    # Submit task if we have at least one file hint
                    if files:
                        from jinx.micro.runtime.api import submit_task as _submit
                        await _submit("quality.scan", files=files[:4], max_ms=600)
                except Exception:
                    pass
            # Self-healing
            if _cooldown_ok(task_name, "heal", 3.0):
                try:
                    _pub_evt("recovery.stage", {"task": task_name, "stage": "heal"})
                except Exception:
                    pass
                try:
                    from jinx.micro.runtime.self_healing import auto_heal_error as _heal
                    await _heal(type(ex).__name__, str(ex), tb)
                except Exception:
                    pass
            # Resilience repairs
            if _cooldown_ok(task_name, "repairs", 15.0):
                try:
                    _pub_evt("recovery.stage", {"task": task_name, "stage": "repairs"})
                except Exception:
                    pass
                try:
                    from jinx.micro.runtime.resilience import schedule_repairs as _rep
                    await _rep()
                except Exception:
                    pass
            # Proactive auto-evolution (improve/mutate/scale) based on failure context
            if _cooldown_ok(task_name, "autoevolve", 30.0):
                try:
                    _pub_evt("recovery.stage", {"task": task_name, "stage": "autoevolve"})
                except Exception:
                    pass
                try:
                    from jinx.micro.runtime.api import submit_task as _submit
                    q = (
                        f"improve robustness, throughput, and fault-tolerance for task '{task_name}' "
                        f"after {type(ex).__name__}: {str(ex)[:200]}"
                    )
                    await _submit("autoevolve.request", query=q)
                except Exception:
                    pass
            # Escalate to self-reprogrammer on repeated failures to mutate the codebase safely
            try:
                # Heuristic: trigger when restart count suggests instability
                rest_n = int(restarts.get(task_name, 0))
                m = metrics.get(task_name)
                cons = int(getattr(m, "consecutive_failures", 0)) if m else 0
            except Exception:
                rest_n = 0
                cons = 0
            if (rest_n >= 2 or cons >= 2) and _cooldown_ok(task_name, "reprogram", 120.0):
                try:
                    _pub_evt("recovery.stage", {"task": task_name, "stage": "reprogram"})
                except Exception:
                    pass
                try:
                    import re as _re, os as _os
                    # Try to extract a local module file from traceback for a focused goal
                    rel_hint = None
                    cwd = (_os.getcwd() or "").strip()
                    if tb:
                        for mm in _re.finditer(r'File "([^"]+\.py)", line (\d+)', tb):
                            ap = str(mm.group(1) or "")
                            if ap and cwd and ap.lower().startswith(cwd.lower()):
                                rel_hint = ap[len(cwd):].lstrip("/\\")
                                break
                    goal = (
                        f"harden and evolve runtime around {rel_hint or task_name}: "
                        f"add RT-safe cancellation handling, optional telemetry no-op, offline fallbacks, "
                        f"admission guards, and micro-modular scaling controls; keep diffs minimal and safe"
                    )
                    from jinx.micro.runtime.api import submit_task as _submit
                    await _submit("reprogram.request", goal=goal, timeout_s=18.0)
                except Exception:
                    pass
            # Self-update on import errors
            if isinstance(ex, (ModuleNotFoundError, ImportError)) and _cooldown_ok(task_name, "selfupdate", 120.0):
                try:
                    _pub_evt("recovery.stage", {"task": task_name, "stage": "selfupdate"})
                except Exception:
                    pass
                try:
                    from jinx.micro.runtime.api import submit_task as _submit
                    await _submit("selfupdate.apply", timeout_s=18.0)
                except Exception:
                    pass

    def _start(name: str) -> None:
        try:
            job_spec = next((j for j in jobs if j.name == name), None)
            if not job_spec:
                return
            try:
                from jinx.observability.otel import span as _span
            except Exception:
                from contextlib import nullcontext as _span  # type: ignore
            with _span(f"supervisor.job_start:{name}"):
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
        try:
            from jinx.observability.otel import span as _span
        except Exception:
            from contextlib import nullcontext as _span  # type: ignore
        with _span("supervisor.loop"):
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
                    # Trigger non-blocking auto-recovery pipeline
                    try:
                        asyncio.create_task(_auto_recover(name or "?", ex))
                    except Exception:
                        pass
                    if not rt.supervise_tasks:
                        continue
                    # Restart/health/backoff logic for this finished task
                    count = restarts.get(name, 0)
                    if count >= rt.autorestart_limit:
                        # Give up on this job
                        m = metrics[name]
                        m.health = JobHealth.DEAD
                        # Critical job => trigger shutdown
                        job_spec = next((j for j in jobs if j.name == name), None)
                        if job_spec and job_spec.critical:
                            try:
                                with _span(f"supervisor.job_dead:{name}"):
                                    pass
                            except Exception:
                                pass
                            try:
                                from jinx.logging_service import bomb_log
                                await bomb_log(f"CRITICAL JOB DEAD: {name} - initiating shutdown")
                            except Exception:
                                pass
                            shutdown_event.set()
                        continue
                    # Rate limiting
                    current_time = time.time()
                    restart_times[name].append(current_time)
                    restart_times[name] = [t for t in restart_times[name] if (current_time - t) < 60.0]
                    restart_rate = len(restart_times[name])
                    job_spec = next((j for j in jobs if j.name == name), None)
                    if job_spec and restart_rate > job_spec.max_restart_rate:
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
                    # Backoff with jitter
                    base = max(1, rt.backoff_min_ms) / 1000.0
                    cap = max(base, rt.backoff_max_ms / 1000.0)
                    delay = min(cap, base * (2 ** count))
                    delay = delay * (0.7 + 0.6 * random.random())
                    if m.health == JobHealth.FAILING:
                        delay *= 2.0
                    await _sleep_cancelable(delay, shutdown_event)
                    if shutdown_event.is_set():
                        break
                    try:
                        with _span(f"supervisor.job_restart:{name}"):
                            _start(name)
                    except Exception:
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
