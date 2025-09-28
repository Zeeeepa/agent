from __future__ import annotations

import asyncio
from typing import Any

from .api import on
from .contracts import PROGRAM_LOG, TASK_PROGRESS, TASK_RESULT, PROGRAM_SPAWN, PROGRAM_EXIT

_started = False


async def _log_handler(topic: str, payload: dict) -> None:
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        pid = str(payload.get("id") or "?")
        name = str(payload.get("name") or "?")
        level = str(payload.get("level") or "info")
        msg = str(payload.get("msg") or "")
        await _append(BLUE_WHISPERS, f"[PROG:{name}:{pid}:{level}] {msg}")
    except Exception:
        pass


async def _task_progress_handler(topic: str, payload: dict) -> None:
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        tid = str(payload.get("id") or "?")
        pct = float(payload.get("pct") or 0.0)
        msg = str(payload.get("msg") or "")
        await _append(BLUE_WHISPERS, f"[TASK:{tid}:progress] {pct:.1f}% {msg}")
    except Exception:
        pass


async def _task_result_handler(topic: str, payload: dict) -> None:
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        tid = str(payload.get("id") or "?")
        ok = bool(payload.get("ok"))
        res = payload.get("result")
        err = str(payload.get("error") or "")
        if ok:
            await _append(BLUE_WHISPERS, f"[TASK:{tid}:ok] {res}")
        else:
            await _append(BLUE_WHISPERS, f"[TASK:{tid}:fail] {err}")
    except Exception:
        pass


async def start_bridge() -> None:
    global _started
    if _started:
        return
    # Subscribe handlers once to mirror runtime activity into logs/observability
    await on(PROGRAM_LOG, _log_handler)
    await on(TASK_PROGRESS, _task_progress_handler)
    await on(TASK_RESULT, _task_result_handler)
    # lightweight spawn/exit echoes
    async def _spawn(topic: str, payload: dict) -> None:
        try:
            from jinx.logger.file_logger import append_line as _append
            from jinx.log_paths import BLUE_WHISPERS
            await _append(BLUE_WHISPERS, f"[PROG:{payload.get('name','?')}:{payload.get('id','?')}:spawn] ready")
        except Exception:
            pass
    async def _exit(topic: str, payload: dict) -> None:
        try:
            from jinx.logger.file_logger import append_line as _append
            from jinx.log_paths import BLUE_WHISPERS
            await _append(BLUE_WHISPERS, f"[PROG:{payload.get('name','?')}:{payload.get('id','?')}:exit] ok={bool(payload.get('ok'))}")
        except Exception:
            pass
    await on(PROGRAM_SPAWN, _spawn)
    await on(PROGRAM_EXIT, _exit)
    _started = True
