from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from .bus import get_bus
from .contracts import PROGRAM_HEARTBEAT, PROGRAM_LOG


class MicroProgram:
    """Base class for autonomous micro-programs controlled by Jinx.

    Subclasses should override `run()` and optionally `on_event()`.
    """

    def __init__(self, name: str | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.name = name or self.__class__.__name__
        self._alive = False
        self._task: asyncio.Task | None = None
        self._hb_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run_wrapper())
        self._hb_task = loop.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        self._alive = False
        if self._task:
            self._task.cancel()
        if self._hb_task:
            self._hb_task.cancel()

    async def _heartbeat_loop(self) -> None:
        bus = get_bus()
        while self._alive:
            await bus.publish(PROGRAM_HEARTBEAT, {"id": self.id, "name": self.name, "ts": time.time()})
            await asyncio.sleep(1.0)

    async def _run_wrapper(self) -> None:
        try:
            await self.run()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self.log(f"crash: {type(e).__name__}: {e}")
        finally:
            self._alive = False

    async def run(self) -> None:
        """Override with the main loop or task body."""
        await asyncio.sleep(0)

    async def on_event(self, topic: str, payload: Any) -> None:
        """Override to handle bus events if subscribed externally."""
        await asyncio.sleep(0)

    async def log(self, msg: str, level: str = "info") -> None:
        await get_bus().publish(PROGRAM_LOG, {"id": self.id, "name": self.name, "level": level, "msg": msg})
