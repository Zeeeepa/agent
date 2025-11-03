from __future__ import annotations

import asyncio
import os
import contextlib
import re as _re
from collections import deque, defaultdict
from jinx.conversation.orchestrator import shatter
from jinx.spinner_service import sigil_spin
import jinx.state as jx_state
from jinx.micro.runtime.task_ctx import current_group


async def frame_shift(q: asyncio.Queue[str]) -> None:
    """Process queue items with bounded concurrency and a single spinner.

    - Schedules up to JINX_FRAME_MAX_CONC conversation steps concurrently.
    - Keeps a single spinner active while there is any work in progress.
    - Preserves cooperative yields and respects shutdown/throttle signals.
    """
    # Concurrency limit for parallel shatter() tasks
    try:
        _MAX_CONC = max(1, int(os.getenv("JINX_FRAME_MAX_CONC", "2")))
    except Exception:
        _MAX_CONC = 2
    # Cap per-group pending queue to avoid unbounded growth
    try:
        _GMAX = max(1, int(os.getenv("JINX_GROUP_PENDING_MAX", "200")))
    except Exception:
        _GMAX = 200

    active: set[asyncio.Task] = set()
    task_group: dict[asyncio.Task, str] = {}
    pending_by_group: dict[str, deque[str]] = defaultdict(deque)
    group_active: set[str] = set()
    groups_rr: list[str] = []
    rr_idx: int = 0

    def _group_of(msg: str) -> str:
        m = _re.match(r"\s*\[#group:([A-Za-z0-9_\-:.]{1,64})\]\s*(.*)", msg or "")
        if m:
            return (m.group(1) or "main").strip().lower() or "main"
        try:
            return (os.getenv("JINX_SESSION", "") or "main").strip() or "main"
        except Exception:
            return "main"

    def _strip_group_tag(msg: str) -> str:
        m = _re.match(r"\s*\[#group:[^\]]+\]\s*(.*)", msg or "")
        return m.group(1) if m else msg
    spin_evt: asyncio.Event | None = None
    spin_task: asyncio.Task | None = None

    async def _ensure_spinner() -> None:
        nonlocal spin_evt, spin_task
        try:
            _spin_on = str(os.getenv("JINX_SPINNER_ENABLE", "1")).strip().lower() not in ("", "0", "false", "off", "no")
        except Exception:
            _spin_on = True
        if not _spin_on:
            return
        if spin_task is None or spin_task.done():
            spin_evt = asyncio.Event()
            spin_task = asyncio.create_task(sigil_spin(spin_evt))

    async def _stop_spinner() -> None:
        nonlocal spin_evt, spin_task
        if spin_evt is not None:
            spin_evt.set()
        if spin_task is not None and not spin_task.done():
            try:
                await asyncio.wait_for(spin_task, timeout=2.0)
            except asyncio.TimeoutError:
                spin_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await spin_task
        spin_evt = None
        spin_task = None

    # Optional per-step hard RT budget (in milliseconds); 0 disables.
    try:
        step_ms = int(os.getenv("JINX_FRAME_STEP_RT_MS", "0"))
    except Exception:
        step_ms = 0

    async def _run_one(s: str, gid: str) -> None:
        try:
            tok = current_group.set(gid)
            if step_ms and step_ms > 0:
                conv_task = asyncio.create_task(shatter(s))
                try:
                    await asyncio.wait_for(conv_task, timeout=step_ms / 1000.0)
                except asyncio.TimeoutError:
                    conv_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await conv_task
            else:
                await shatter(s)
        except Exception:
            # shatter() logs internally; swallow here to keep scheduler alive
            pass
        finally:
            try:
                current_group.reset(tok)  # type: ignore[name-defined]
            except Exception:
                pass

    try:
        while True:
            # Respect global shutdown fast-path
            if jx_state.shutdown_event.is_set():
                break
            # Soft-throttle: pause intake while the system is saturated
            while jx_state.throttle_event.is_set():
                if jx_state.shutdown_event.is_set():
                    break
                await asyncio.sleep(0.05)
            if jx_state.shutdown_event.is_set():
                break

            # Drain inbound queue briefly into per-group pending buffers
            drained = 0
            while drained < 8:  # cap per loop to avoid starvation
                try:
                    item = await asyncio.wait_for(q.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    break
                gid = _group_of(item)
                if gid not in pending_by_group:
                    pending_by_group[gid] = deque(maxlen=_GMAX)
                    groups_rr.append(gid)
                pending_by_group[gid].append(item)
                drained += 1

            # Fill up to concurrency limit with fair round-robin across groups
            filled = False
            if groups_rr:
                n = len(groups_rr)
                start = rr_idx % max(1, n)
                i = 0
                while len(active) < _MAX_CONC and i < n:
                    gid = groups_rr[(start + i) % n]
                    i += 1
                    if gid in group_active:
                        continue
                    dq = pending_by_group.get(gid)
                    if not dq:
                        continue
                    raw = dq.popleft()
                    if not dq:
                        # keep group in RR list but no pending now; it may receive items later
                        pass
                    msg = _strip_group_tag(raw)
                    await _ensure_spinner()
                    t = asyncio.create_task(_run_one(msg, gid))
                    active.add(t)
                    task_group[t] = gid
                    group_active.add(gid)
                    filled = True
                rr_idx = (start + i) % max(1, len(groups_rr))

            # Reap any finished tasks
            done_now = [t for t in active if t.done()]
            for t in done_now:
                active.discard(t)
                with contextlib.suppress(Exception):
                    _ = t.result()
                # release per-group slot
                try:
                    gid = task_group.pop(t, None)
                    if gid is not None and gid in group_active:
                        group_active.discard(gid)
                except Exception:
                    pass

            # Prune empty groups from RR when they have no pending and are not active
            if groups_rr:
                new_rr = []
                for g in groups_rr:
                    dq = pending_by_group.get(g)
                    if (dq and len(dq) > 0) or (g in group_active):
                        new_rr.append(g)
                    else:
                        # drop empty group from maps
                        pending_by_group.pop(g, None)
                groups_rr = new_rr
                if rr_idx >= len(groups_rr):
                    rr_idx = 0

            # If nothing to do, wait briefly or for shutdown
            if not active and not filled:
                # No in-flight work; stop spinner if running
                await _stop_spinner()
                # Wait for either new item or shutdown
                get_task = asyncio.create_task(q.get())
                shut_task = asyncio.create_task(jx_state.shutdown_event.wait())
                done, _ = await asyncio.wait({get_task, shut_task}, return_when=asyncio.FIRST_COMPLETED)
                if shut_task in done:
                    if not get_task.done():
                        get_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await get_task
                    break
                # Got a new item; enqueue into group buffer and continue loop
                try:
                    item2 = get_task.result()
                except Exception:
                    continue
                gid2 = _group_of(item2)
                if gid2 not in pending_by_group:
                    pending_by_group[gid2] = deque(maxlen=_GMAX)
                    groups_rr.append(gid2)
                pending_by_group[gid2].append(item2)
            else:
                # Give control back to event loop to keep RT responsiveness
                await asyncio.sleep(0)
    finally:
        # Stop spinner and cancel any remaining tasks on exit
        with contextlib.suppress(Exception):
            await _stop_spinner()
        for t in list(active):
            if not t.done():
                t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*active, return_exceptions=True)
