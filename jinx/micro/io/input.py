"""Interactive input micro-module.

Provides an async prompt using prompt_toolkit and pushes sanitized user input
into an asyncio queue. Includes an inactivity watchdog that emits
"<no_response>" after a configurable timeout to keep the agent responsive.
"""

from __future__ import annotations

import asyncio
import importlib
import contextlib
from jinx.bootstrap import ensure_optional
from typing import Any, cast
from jinx.state import boom_limit
from jinx.logging_service import blast_mem, bomb_log
from jinx.log_paths import TRIGGER_ECHOES, BLUE_WHISPERS
from jinx.async_utils.queue import try_put_nowait
import jinx.state as jx_state


# Ensure prompt_toolkit is present at import time to avoid installing in an active event loop
ensure_optional(["prompt_toolkit"])  # installs if missing


async def neon_input(qe: asyncio.Queue[str]) -> None:
    """Read user input and feed it into the provided queue.

    Parameters
    ----------
    qe : asyncio.Queue[str]
        Target queue for sanitized user input.
    """
    # Lazily import prompt_toolkit symbols
    _ptk = importlib.import_module("prompt_toolkit")
    _ptk_keys = importlib.import_module("prompt_toolkit.key_binding")
    PromptSession = getattr(_ptk, "PromptSession")
    KeyBindings = getattr(_ptk_keys, "KeyBindings")

    finger_wire = KeyBindings()
    sess = PromptSession(key_bindings=finger_wire)
    boom_clock: dict[str, float] = {"time": asyncio.get_running_loop().time()}
    activity = asyncio.Event()

    @finger_wire.add("<any>")
    def _(triggerbit) -> None:  # prompt_toolkit callback
        boom_clock["time"] = asyncio.get_running_loop().time()
        triggerbit.app.current_buffer.insert_text(triggerbit.key_sequence[0].key)
        # Signal activity to reset the inactivity timer immediately
        activity.set()

    async def kaboom_watch() -> None:
        """Emit <no_response> after inactivity using a reactive timer.

        Avoids periodic polling by waiting for either activity or timeout.
        """
        while True:
            # Calculate remaining time based on last activity
            now = asyncio.get_running_loop().time()
            remaining = max(0.0, boom_limit - (now - boom_clock["time"]))
            activity.clear()
            try:
                # Wait for either new activity or the inactivity timeout
                await asyncio.wait_for(activity.wait(), timeout=remaining)
                # Activity occurred: loop to recalculate remaining
                continue
            except asyncio.TimeoutError:
                # Timeout: no activity within boom_limit
                await blast_mem("<no_response>")
                await bomb_log("<no_response>", TRIGGER_ECHOES)
                # Do not disrupt FIFO order: emit only if queue has space
                placed = try_put_nowait(qe, "<no_response>")
                if not placed:
                    await bomb_log("<no_response> skipped: input queue saturated", BLUE_WHISPERS)
                boom_clock["time"] = asyncio.get_running_loop().time()

    watch_task = asyncio.create_task(kaboom_watch())
    try:
        prompt_task: asyncio.Task[str] | None = None
        shutdown_task: asyncio.Task[bool] | None = None  # Event.wait() resolves to True
        while True:
            try:
                # Race the prompt against a shutdown signal to exit promptly
                pt: asyncio.Task[str] = asyncio.create_task(sess.prompt_async("\n"))
                # Ensure any exception (including BaseException like KeyboardInterrupt) is retrieved
                def _swallow_task_exc(t: asyncio.Task) -> None:
                    try:
                        # Using result() to also re-raise BaseException subclasses
                        _ = t.result()
                    except BaseException:
                        pass
                pt.add_done_callback(_swallow_task_exc)
                st: asyncio.Task[bool] = asyncio.create_task(jx_state.shutdown_event.wait())
                tasks: set[asyncio.Future[Any]] = {cast(asyncio.Future[Any], pt), cast(asyncio.Future[Any], st)}
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if st in done:
                    # Politely ask prompt_toolkit to exit instead of cancelling the task
                    try:
                        sess.app.exit(exception=EOFError())
                    except Exception:
                        pass
                    # Ensure prompt_task finishes and swallow any BaseException (e.g., KeyboardInterrupt)
                    with contextlib.suppress(BaseException):
                        await pt
                    break
                # Got user input
                st.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await st
                v: str = pt.result()
                if v.strip():
                    await bomb_log(v, TRIGGER_ECHOES)
                    await qe.put(v.strip())
                # expose tasks for final cleanup
                prompt_task = pt
                shutdown_task = st
            except (EOFError, KeyboardInterrupt):
                # Treat both as a clean exit of input loop
                if prompt_task is not None:
                    with contextlib.suppress(BaseException):
                        await prompt_task
                break
            except Exception as e:  # pragma: no cover - guard rail for TTY issues
                await bomb_log(f"ERROR INPUT chaos keys went rogue: {e}")
    finally:
        # Ensure watchdog is cancelled when input loop exits
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        # Force-close prompt_toolkit app and consume any leftover tasks
        with contextlib.suppress(Exception):
            sess.app.exit(exception=EOFError())
        if prompt_task is not None:
            pt2 = cast(asyncio.Task[Any], prompt_task)
            if not pt2.done():
                with contextlib.suppress(BaseException):
                    await pt2
        if shutdown_task is not None:
            st2 = cast(asyncio.Task[Any], shutdown_task)
            if not st2.done():
                st2.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await st2
