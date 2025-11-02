from __future__ import annotations

import asyncio
import os
from typing import Callable, Awaitable, Dict, Set

from jinx.log_paths import SANDBOX_DIR, TRIGGER_ECHOES
import time
from jinx.async_utils.rt import rt_section

# Max lines per second per sandbox file to process (others dropped). Tunable via env.
_SANDBOX_MAX_LPS = int(os.getenv("EMBED_SANDBOX_MAX_LPS", "5"))
# Trigger echoes enable + rate
_TRIG_ENABLE = os.getenv("EMBED_TRIG_ENABLE", "1").strip().lower() not in ("", "0", "false", "off", "no")
_TRIG_MAX_LPS = int(os.getenv("EMBED_TRIG_MAX_LPS", "12"))
# Preroll config: how many existing lines to ingest on startup per file, and max bytes read
_PREROLL_LINES = int(os.getenv("EMBED_SANDBOX_PREROLL_LINES", "50"))
_PREROLL_MAX_BYTES = int(os.getenv("EMBED_SANDBOX_PREROLL_MAX_BYTES", str(256 * 1024)))

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
    if _TRIG_ENABLE:
        tasks.append(asyncio.create_task(_watch_trigger(cb)))
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _ensure_paths() -> None:
    # Ensure parent dirs so tailers can open files reliably
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    # Ensure trigger echoes file exists (best-effort); tailer will create if missing
    try:
        base = os.path.dirname(TRIGGER_ECHOES)
        if base:
            os.makedirs(base, exist_ok=True)
        if not os.path.exists(TRIGGER_ECHOES):
            with open(TRIGGER_ECHOES, "a", encoding="utf-8") as _:
                pass
    except Exception:
        pass


def _read_tail_lines(path: str, max_bytes: int, max_lines: int) -> list[str]:
    """Read up to the last `max_lines` lines from the file, scanning at most `max_bytes`.

    Best-effort: if file is smaller than `max_bytes`, reads once; otherwise reads the tail chunk.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    start = max(0, size - max_bytes)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if start > 0:
                f.seek(start)
                # Skip partial line if we started mid-line
                f.readline()
            data = f.read()
    except OSError:
        return []
    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
    if max_lines > 0:
        lines = lines[-max_lines:]
    return lines


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
        # Optional preroll: embed last N existing lines before switching to live tail
        if _PREROLL_LINES > 0:
            for ln in _read_tail_lines(path, _PREROLL_MAX_BYTES, _PREROLL_LINES):
                # Reuse same callback and limiter semantics
                await cb(ln, source, "line")
        f.seek(0, os.SEEK_END)
        # Rate limiting state per file
        win_start = time.time()
        processed = 0
        budget_ms = int(os.getenv("JINX_HARD_RT_BUDGET_MS", "40"))
        async with rt_section(budget_ms) as rt:
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.2)
                    await rt.tick()
                    continue
                line = line.strip()
                if line:
                    # Apply simple token bucket for sandbox sources
                    if source.startswith("sandbox/") and _SANDBOX_MAX_LPS > 0:
                        now = time.time()
                        if now - win_start >= 1.0:
                            win_start = now
                            processed = 0
                        if processed >= _SANDBOX_MAX_LPS:
                            # Drop extra lines in the current 1s window
                            await rt.tick()
                            continue
                        processed += 1
                    if source == "trigger_echoes" and _TRIG_MAX_LPS > 0:
                        now = time.time()
                        if now - win_start >= 1.0:
                            win_start = now
                            processed = 0
                        if processed >= _TRIG_MAX_LPS:
                            await rt.tick()
                            continue
                        processed += 1
                    await cb(line, source, "line")
                await rt.tick()


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
        budget_ms = int(os.getenv("JINX_HARD_RT_BUDGET_MS", "40"))
        async with rt_section(budget_ms) as rt:
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
                        await rt.tick()
                        continue
                    if p not in known or known[p] != mtime:
                        known[p] = mtime
                        await spawn_tail(p)
                    await rt.tick()

                # Reap finished
                dead = [k for k, t in tasks.items() if t.done()]
                for k in dead:
                    tasks.pop(k, None)
                    tailed.discard(k)
                await asyncio.sleep(0.7)
                await rt.tick()
    finally:
        for t in tasks.values():
            t.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


async def _watch_trigger(cb: Callback) -> None:
    # Single-file tailer for trigger echoes
    # Slightly lower preroll for triggers by reusing _read_tail_lines directly
    try:
        # Preroll
        for ln in _read_tail_lines(TRIGGER_ECHOES, _PREROLL_MAX_BYTES, min(_PREROLL_LINES, 20)):
            await cb(ln, "trigger_echoes", "line")
    except Exception:
        pass
    # Then switch to live tail
    await _tail_file(TRIGGER_ECHOES, source="trigger_echoes", cb=cb)
