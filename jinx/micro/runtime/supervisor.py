from __future__ import annotations

import asyncio
import os
import time
from typing import Dict

from .bus import get_bus
from .contracts import PROGRAM_HEARTBEAT, PROGRAM_EXIT, PROGRAM_SPAWN
from .registry import get_registry


class Supervisor:
    def __init__(self) -> None:
        self._hb: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._watch_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._watch_task:
            return
        bus = get_bus()
        await bus.subscribe(PROGRAM_HEARTBEAT, self._on_hb)
        loop = asyncio.get_running_loop()
        self._watch_task = loop.create_task(self._watchdog())

    async def _on_hb(self, topic: str, payload: dict) -> None:
        pid = str(payload.get("id") or "")
        if not pid:
            return
        async with self._lock:
            self._hb[pid] = float(payload.get("ts") or time.time())

    async def _watchdog(self) -> None:
        try:
            ttl = float(os.getenv("JINX_RUNTIME_HEARTBEAT_SEC", "5"))
        except Exception:
            ttl = 5.0
        while True:
            try:
                now = time.time()
                stale: list[str] = []
                async with self._lock:
                    for pid, ts in list(self._hb.items()):
                        if (now - ts) > ttl:
                            stale.append(pid)
                            self._hb.pop(pid, None)
                # Announce exits for stale programs (best-effort)
                for pid in stale:
                    await get_bus().publish(PROGRAM_EXIT, {"id": pid, "name": "?", "ok": False})
            except Exception:
                pass
            await asyncio.sleep(ttl / 2.0)


_supervisor: Supervisor | None = None


def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor()
    return _supervisor
