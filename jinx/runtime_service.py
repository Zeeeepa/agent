"""Async runtime loop for the Jinx agent.

``pulse_core()`` wires input, spinner, and conversation services together and
drives the primary event loop. It is intentionally lightweight, cancellation
friendly, and side-effect contained for testability.
"""

from __future__ import annotations

import asyncio
from jinx.banner_service import show_banner
import concurrent.futures as _cf
from jinx.utils import chaos_patch
from jinx.runtime import start_input_task, frame_shift as _frame_shift
from jinx.embeddings import start_embeddings_task, start_project_embeddings_task
import jinx.state as jx_state
import contextlib
from jinx.memory.optimizer import stop as stop_memory_optimizer, start_memory_optimizer_task
from jinx.conversation.error_worker import stop_error_worker
from jinx.settings import Settings
from jinx.supervisor import run_supervisor, SupervisedJob
from jinx.priority import start_priority_dispatcher_task
from jinx.autotune import start_autotune_task
from jinx.watchdog import start_watchdog_task

async def pulse_core(settings: Settings | None = None) -> None:
    """Run the main asynchronous processing loop.

    The loop:
    - Shows the startup banner.
    - Starts an input task that feeds a queue with user messages.
    - Optionally dispatches through a priority-aware relay to the frame processor.
    - Displays a spinner per message while executing the conversation step.
    - Supervises background tasks with auto-restart and graceful shutdown.
    """
    show_banner()

    # Resolve settings and apply compatibility state
    cfg = settings or Settings.from_env()
    cfg.apply_to_state()
    # Minimal startup summary to stdout (no CLI required)
    try:
        print(
            f"‖ Auto-tune: prio={'on' if cfg.runtime.use_priority_queue else 'off'}, "
            f"threads={cfg.runtime.threads_max_workers}, "
            f"queue={cfg.runtime.queue_maxsize}, rt={cfg.runtime.hard_rt_budget_ms}ms"
        )
    except Exception:
        pass

    # Bounded queues to avoid unbounded memory growth under bursts
    q_in: asyncio.Queue[str] = asyncio.Queue(maxsize=cfg.runtime.queue_maxsize)
    q_proc: asyncio.Queue[str] = asyncio.Queue(maxsize=cfg.runtime.queue_maxsize)

    # Always route through the dispatcher so autotune can toggle priority dynamically
    q_for_frame = q_proc

    async with chaos_patch():
        # Configure default thread pool for to_thread operations
        try:
            loop = asyncio.get_running_loop()
            loop.set_default_executor(_cf.ThreadPoolExecutor(max_workers=cfg.runtime.threads_max_workers, thread_name_prefix="jinx-worker"))
        except Exception:
            pass
        # Compose supervised jobs
        job_specs: list[SupervisedJob] = [
            SupervisedJob(name="input", start=lambda: start_input_task(q_in)),
            SupervisedJob(name="frame", start=lambda: asyncio.create_task(_frame_shift(q_for_frame))),
            SupervisedJob(name="priority", start=lambda: start_priority_dispatcher_task(q_in, q_proc, cfg)),
            SupervisedJob(name="embeddings", start=lambda: start_embeddings_task()),
            SupervisedJob(name="memopt", start=lambda: start_memory_optimizer_task()),
            SupervisedJob(name="proj-embed", start=lambda: start_project_embeddings_task()),
            SupervisedJob(name="autotune", start=lambda: start_autotune_task(q_in, cfg)),
            SupervisedJob(name="watchdog", start=lambda: start_watchdog_task(cfg)),
        ]

        try:
            # Run supervised set; returns when shutdown_event is set
            await run_supervisor(job_specs, jx_state.shutdown_event, cfg)
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Signal global shutdown to all components
            jx_state.shutdown_event.set()
        finally:
            # Stop auxiliary background workers first
            with contextlib.suppress(Exception):
                await stop_error_worker()
                await stop_memory_optimizer()
