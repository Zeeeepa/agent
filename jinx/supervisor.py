from __future__ import annotations

import asyncio
import random
import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from jinx.settings import Settings


@dataclass(slots=True)
class SupervisedJob:
    name: str
    start: Callable[[], "asyncio.Task[None]"]


async def _sleep_cancelable(delay: float, cancel_event: asyncio.Event) -> None:
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=delay)
    except asyncio.TimeoutError:
        return


async def run_supervisor(jobs: list[SupervisedJob], shutdown_event: asyncio.Event, settings: Settings) -> None:
    """
    Run and supervise background jobs with auto-restart and bounded backoff.

    - Restarts tasks on unexpected failure up to `autorestart_limit` times.
    - Applies jittered exponential backoff within [backoff_min_ms, backoff_max_ms].
    - Cooperates with `shutdown_event` for prompt termination.
    """
    rt = settings.runtime
    tasks: Dict[str, asyncio.Task[None]] = {}
    restarts: Dict[str, int] = {}

    def _start(name: str) -> None:
        try:
            t = next(j.start for j in jobs if j.name == name)()
        except StopIteration:
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
                    continue
                if not rt.supervise_tasks:
                    continue
                count = restarts.get(name, 0)
                if count >= rt.autorestart_limit:
                    # Give up on this job
                    continue
                restarts[name] = count + 1
                # Compute backoff with jitter
                base = max(1, rt.backoff_min_ms) / 1000.0
                cap = max(base, rt.backoff_max_ms / 1000.0)
                delay = min(cap, base * (2 ** count))
                delay = delay * (0.7 + 0.6 * random.random())
                # Wait unless shutdown requested
                await _sleep_cancelable(delay, shutdown_event)
                if shutdown_event.is_set():
                    break
                _start(name)
    finally:
        # cancel all active
        for t in tasks.values():
            t.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
