"""Service contracts (protocols) for the Jinx agent.

This module defines lightweight Protocols describing the behavior of
collaborating services. It enables dependency injection and easier unit testing
without enforcing a specific implementation. Implementers should satisfy these
contracts to be interchangeable in the runtime.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Protocol, runtime_checkable
from jinx.log_paths import BLUE_WHISPERS
import asyncio


@runtime_checkable
class InputPort(Protocol):
    """Contract for user input providers.

    Implementations should push sanitized user messages into the provided
    asyncio.Queue. Implementations must be cancellation-friendly and not leak
    background tasks upon shutdown.
    """

    async def __call__(self, queue: asyncio.Queue[str]) -> None:  # pragma: no cover - protocol signature
        ...


@runtime_checkable
class SpinnerPort(Protocol):
    """Contract for progress indicator that runs until an event is set."""

    async def __call__(self, stop_event: asyncio.Event) -> None:  # pragma: no cover
        ...


@runtime_checkable
class ConversationPort(Protocol):
    """Contract for LLM-driven conversation step executor."""

    async def shatter(self, prompt: str, err: str | None = None) -> None:  # pragma: no cover
        ...

    async def corrupt_report(self, err: str | None) -> None:  # pragma: no cover
        ...


@runtime_checkable
class LoggerPort(Protocol):
    """Contract for structured logging helpers."""

    async def glitch_pulse(self) -> str:  # pragma: no cover
        ...

    async def blast_mem(self, x: str, n: int = 500) -> None:  # pragma: no cover
        ...

    async def bomb_log(self, t: str, bin: str = BLUE_WHISPERS) -> None:  # pragma: no cover
        ...


@runtime_checkable
class BannerPort(Protocol):
    """Contract for startup banner renderers."""

    def show_banner(self) -> None:  # pragma: no cover
        ...
