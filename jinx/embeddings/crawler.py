from __future__ import annotations

import asyncio
import os
from typing import Callable, Awaitable, Dict, Set

from jinx.log_paths import SANDBOX_DIR

Callback = Callable[[str, str, str], Awaitable[None]]  # (text, source, kind)


async def start_realtime_collection(cb: Callback) -> None:
    """Start realtime collectors for trigger_echoes and sandbox logs.

    - Tails all *.log files inside log/sandbox/
    - Discovers new sandbox logs periodically
    """
    await _ensure_paths()

    tasks = [
        asyncio.create_task(_watch_sandbox(cb)),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _ensure_paths() -> None:
    # Ensure parent dirs so tailers can open files reliably
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    # No trigger_echoes ingestion


async def _tail_file(path: str, *, source: str, cb: Callback) -> None:
    # Start from EOF; only new lines are processed
    try:
        f = open(path, "r", encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        # If deleted later, re-create
        while True:
            await asyncio.sleep(0.5)
            try:
                f = open(path, "r", encoding="utf-8", errors="ignore")
                break
            except FileNotFoundError:
                continue

    with f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.2)
                continue
            line = line.strip()
            if line:
                await cb(line, source, "line")


async def _watch_sandbox(cb: Callback) -> None:
    known: Dict[str, int] = {}
    tailed: Set[str] = set()
    tasks: Dict[str, asyncio.Task] = {}

    async def spawn_tail(p: str) -> None:
        if p in tailed:
            return
        tailed.add(p)
        tasks[p] = asyncio.create_task(_tail_file(p, source=f"sandbox/{os.path.basename(p)}", cb=cb))

    try:
        while True:
            try:
                entries = [
                    os.path.join(SANDBOX_DIR, x)
                    for x in os.listdir(SANDBOX_DIR)
                    if x.lower().endswith(".log")
                ]
            except FileNotFoundError:
                entries = []

            for p in entries:
                try:
                    st = os.stat(p)
                    mtime = int(st.st_mtime)
                except FileNotFoundError:
                    continue
                if p not in known or known[p] != mtime:
                    known[p] = mtime
                    await spawn_tail(p)

            # Reap finished
            dead = [k for k, t in tasks.items() if t.done()]
            for k in dead:
                tasks.pop(k, None)
                tailed.discard(k)

            await asyncio.sleep(0.7)
    finally:
        for t in tasks.values():
            t.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
