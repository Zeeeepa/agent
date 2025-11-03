from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any, Awaitable, Callable, Optional

from .bus import get_bus
from .registry import get_registry
from .supervisor import get_supervisor
from .contracts import TASK_REQUEST, TASK_RESULT, TASK_PROGRESS, PROGRAM_SPAWN
from .program import MicroProgram
from jinx.micro.llm.macro_registry import register_macro as _register_macro
from jinx.micro.net.client import prewarm_openai_client as _prewarm_openai


_bridge_started: bool = False
_selfstudy_started: bool = False
_prewarmed_openai: bool = False
_bg_tasks: list[asyncio.Task] = []


async def ensure_runtime() -> None:
    """Ensure core runtime is up and start self-study components once.

    Components:
    - Supervisor watchdog (heartbeats)
    - Bridge to log program/task events
    - Project embeddings service (code indexer)
    - Realtime embeddings service (sandbox/log tailing)
    """
    global _bridge_started, _selfstudy_started, _bg_tasks
    await get_supervisor().start()
    # One-time OpenAI HTTP pool prewarm (default enabled)
    global _prewarmed_openai
    if not _prewarmed_openai:
        try:
            on = str(os.getenv("JINX_OPENAI_PREWARM", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            on = True
        if on:
            try:
                _prewarm_openai()
            except Exception:
                pass
        _prewarmed_openai = True
    # Start bridge once
    if not _bridge_started:
        try:
            # Lazy import to avoid circulars
            from .bridge import start_bridge  # type: ignore
            await start_bridge()
            _bridge_started = True
        except Exception:
            _bridge_started = True  # prevent repeated attempts
    # (prewarm already handled above)

    # Start self-study (embeddings services) once
    if not _selfstudy_started:
        try:
            # Project code embeddings service
            from jinx.micro.embeddings.project_service import start_project_embeddings_task  # type: ignore
            t1 = start_project_embeddings_task(root=None)
            _bg_tasks.append(t1)
        except Exception:
            pass
        try:
            # Realtime log tailing embeddings service
            from jinx.micro.embeddings.service import start_embeddings_task  # type: ignore
            t2 = start_embeddings_task()
            _bg_tasks.append(t2)
        except Exception:
            pass
        # Background symbol indexer (env-gated, default ON)
        try:
            if str(os.getenv("JINX_SYMBOL_INDEX_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no"):
                try:
                    from jinx.micro.embeddings.symbol_indexer import start_symbol_indexer_task as _start_symidx  # local import to avoid cycles
                    t3 = _start_symidx()
                    _bg_tasks.append(t3)
                except Exception:
                    pass
        except Exception:
            pass
        # Auto-fix program (env-gated, default ON)
        try:
            if str(os.getenv("JINX_AUTOFIX_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no"):
                try:
                    from jinx.micro.runtime.auto_fix_program import spawn_autofix as _spawn_autofix  # local import to avoid cycles
                    await _spawn_autofix()
                except Exception:
                    pass
        except Exception:
            pass
        # Repair program (env-gated, default ON)
        try:
            if str(os.getenv("JINX_REPAIR_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no"):
                try:
                    from jinx.micro.runtime.repair_program import spawn_repair as _spawn_repair  # local import
                    await _spawn_repair()
                except Exception:
                    pass
        except Exception:
            pass
        _selfstudy_started = True
    # Schedule repairs for any missing modules detected by resilience hook (best-effort)
    try:
        from jinx.micro.runtime.resilience import schedule_repairs as _sched_repairs  # local import
        await _sched_repairs()
    except Exception:
        pass
    # Start resilience monitor to periodically resubmit repairs
    try:
        from jinx.micro.runtime.resilience import start_resilience_monitor_task as _start_resmon  # local import
        tmon = _start_resmon()
        _bg_tasks.append(tmon)
    except Exception:
        pass


async def stop_selfstudy() -> None:
    """Cancel and await background self-study tasks started by ensure_runtime()."""
    global _bg_tasks
    if not _bg_tasks:
        return
    tasks = list(_bg_tasks)
    _bg_tasks = []
    # Cancel any still-running tasks
    for t in tasks:
        if t and not t.done():
            t.cancel()
    # Await tasks to retrieve exceptions and avoid pending-task warnings
    with contextlib.suppress(Exception):
        await asyncio.gather(*[t for t in tasks if t is not None], return_exceptions=True)

# Pub/Sub
async def on(topic: str, handler: Callable[[str, Any], Awaitable[None]]) -> None:
    await get_bus().subscribe(topic, handler)


async def emit(topic: str, payload: Any) -> None:
    await get_bus().publish(topic, payload)


# Program control API (programs are arbitrary Python objects — typically MicroProgram)
async def register_program(pid: str, prog: object) -> None:
    await get_registry().put(pid, prog)
    await emit(PROGRAM_SPAWN, {"id": pid, "name": getattr(prog, "name", prog.__class__.__name__)})


async def get_program(pid: str) -> Optional[object]:
    return await get_registry().get(pid)


async def remove_program(pid: str) -> None:
    await get_registry().remove(pid)


# Convenience wrappers
async def spawn(program: MicroProgram) -> str:
    """Start and register a MicroProgram; return its id."""
    await program.start()
    await register_program(program.id, program)
    return program.id


async def stop(pid: str) -> None:
    prog = await get_program(pid)
    if hasattr(prog, "stop"):
        try:
            await prog.stop()  # type: ignore[func-returns-value]
        except Exception:
            pass
    await remove_program(pid)


async def list_programs() -> list[str]:
    return await get_registry().list_ids()


# Prompt macro API (for microprograms to extend prompt composition)
async def register_prompt_macro(namespace: str, handler) -> None:
    """Register a dynamic prompt macro provider.

    Handler signature: async def handler(args: List[str], ctx: Any) -> str
    """
    await _register_macro(namespace, handler)


# Task APIs
async def submit_task(name: str, *args: Any, **kwargs: Any) -> str:
    import uuid
    tid = uuid.uuid4().hex[:12]
    await emit(TASK_REQUEST, {"id": tid, "name": name, "args": args, "kwargs": kwargs})
    return tid


async def report_progress(tid: str, pct: float, msg: str) -> None:
    await emit(TASK_PROGRESS, {"id": tid, "pct": float(pct), "msg": str(msg)})


async def report_result(tid: str, ok: bool, result: Any = None, error: str | None = None) -> None:
    await emit(TASK_RESULT, {"id": tid, "ok": bool(ok), "result": result, "error": error})
