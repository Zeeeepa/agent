from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from .bus import get_bus
from .registry import get_registry
from .supervisor import get_supervisor
from .contracts import TASK_REQUEST, TASK_RESULT, TASK_PROGRESS, PROGRAM_SPAWN
from .program import MicroProgram
from jinx.micro.llm.macro_registry import register_macro as _register_macro


async def ensure_runtime() -> None:
    await get_supervisor().start()


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
