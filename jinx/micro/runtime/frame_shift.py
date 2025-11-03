from __future__ import annotations

import asyncio
import os
import contextlib
import re as _re
import time
from collections import deque, defaultdict
from typing import Sequence
from jinx.conversation.orchestrator import shatter
from jinx.spinner_service import sigil_spin
import jinx.state as jx_state
from jinx.micro.runtime.task_ctx import current_group
from jinx.micro.rt.backpressure import clear_throttle_if_ttl
from jinx.micro.runtime.plugins import publish_event as _publish_event


# Advanced multi-message splitting with semantic awareness
def _split_if_multi(msg: str, group_id: str) -> Sequence[str] | None:
    """Split a message into multiple sub-requests if it contains multiple distinct tasks.
    
    Uses advanced pattern recognition to identify:
    - Numbered lists (1., 2., 3. or 1), 2), 3))
    - Bullet points (-, *, •)
    - Semicolon-separated commands
    - Newline-separated short instructions
    
    Returns None if message should not be split, otherwise returns list of sub-messages.
    """
    try:
        _MAX_SPLIT = max(2, int(os.getenv("JINX_MULTI_SPLIT_MAX", "6")))
    except Exception:
        _MAX_SPLIT = 6
        
    try:
        _SPLIT_ON = str(os.getenv("JINX_MULTI_SPLIT_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        _SPLIT_ON = True
        
    if not _SPLIT_ON or not msg or len(msg.strip()) < 10:
        return None
        
    stripped = msg.strip()
    
    # Pattern 1: Numbered list with periods (1. Task one 2. Task two)
    numbered_period = _re.findall(r'(?:^|\n)\s*\d+\.\s+([^\n]+)', stripped)
    if len(numbered_period) >= 2 and len(numbered_period) <= _MAX_SPLIT:
        return [s.strip() for s in numbered_period if s.strip()]
    
    # Pattern 2: Numbered list with parentheses (1) Task one 2) Task two)
    numbered_paren = _re.findall(r'(?:^|\n)\s*\d+\)\s+([^\n]+)', stripped)
    if len(numbered_paren) >= 2 and len(numbered_paren) <= _MAX_SPLIT:
        return [s.strip() for s in numbered_paren if s.strip()]
    
    # Pattern 3: Bullet points (- item, * item, • item)
    bullets = _re.findall(r'(?:^|\n)\s*[-*•]\s+([^\n]+)', stripped)
    if len(bullets) >= 2 and len(bullets) <= _MAX_SPLIT:
        return [s.strip() for s in bullets if s.strip()]
    
    # Pattern 4: Semicolon-separated commands (short ones only)
    if ';' in stripped and stripped.count(';') <= _MAX_SPLIT:
        parts = [p.strip() for p in stripped.split(';') if p.strip()]
        # Only split if all parts are relatively short (likely commands)
        if 2 <= len(parts) <= _MAX_SPLIT and all(len(p) < 150 for p in parts):
            return parts
    
    # Pattern 5: Newline-separated short instructions
    lines = [ln.strip() for ln in stripped.split('\n') if ln.strip()]
    if 2 <= len(lines) <= _MAX_SPLIT:
        # Check if they look like short commands (all under 100 chars)
        if all(len(ln) < 100 and not any(kw in ln.lower() for kw in ['however', 'therefore', 'because', 'although']) for ln in lines):
            return lines
    
    return None


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
    # Per-group concurrency cap (how many tasks from the same group can run concurrently)
    try:
        _GCONC = max(1, int(os.getenv("JINX_GROUP_MAX_CONC", "2")))
    except Exception:
        _GCONC = 2
    # Max items to drain from inbound queue per loop iteration
    try:
        _DRAIN_MAX = max(1, int(os.getenv("JINX_FRAME_DRAIN_MAX", "16")))
    except Exception:
        _DRAIN_MAX = 16
    # How many pending items per group to peek for simple-locator preference (0 = only head)
    try:
        _SIMPLE_SCAN = max(0, int(os.getenv("JINX_SIMPLE_LOCATOR_SCAN", "0")))
    except Exception:
        _SIMPLE_SCAN = 0
    # Threshold for embeddings-based locator classifier margin (pos - neg)
    try:
        _LOC_THRESH = float(os.getenv("JINX_LOCATOR_THRESH", "0.06"))
    except Exception:
        _LOC_THRESH = 0.06
    # Multi-split of a single inbound message into sub-requests
    try:
        _SPLIT_ON = str(os.getenv("JINX_MULTI_SPLIT_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        _SPLIT_ON = True
    try:
        _SPLIT_MAX = max(2, int(os.getenv("JINX_MULTI_SPLIT_MAX", "6")))
    except Exception:
        _SPLIT_MAX = 6

    active: set[asyncio.Task] = set()
    task_group: dict[asyncio.Task, str] = {}
    pending_by_group: dict[str, deque[str]] = defaultdict(deque)
    group_active_count: dict[str, int] = {}
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
    spin_start_t: float | None = None

    async def _ensure_spinner() -> None:
        nonlocal spin_evt, spin_task, spin_start_t
        try:
            _spin_on = str(os.getenv("JINX_SPINNER_ENABLE", "1")).strip().lower() not in ("", "0", "false", "off", "no")
        except Exception:
            _spin_on = True
        if not _spin_on:
            return
        if spin_task is None or spin_task.done():
            spin_evt = asyncio.Event()
            spin_task = asyncio.create_task(sigil_spin(spin_evt))
            # Publish spinner start
            try:
                spin_start_t = time.perf_counter()
                _publish_event("spinner.start", {"t": spin_start_t})
            except Exception:
                pass

    async def _stop_spinner() -> None:
        nonlocal spin_evt, spin_task, spin_start_t
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
        # Publish spinner stop with duration
        try:
            t1 = time.perf_counter()
            dt = (t1 - spin_start_t) if spin_start_t else 0.0
            _publish_event("spinner.stop", {"dt": dt})
        except Exception:
            pass
        spin_start_t = None
        # Ensure throttle is cleared when spinner stops to allow intake to resume
        try:
            clear_throttle_if_ttl()
        except Exception:
            pass
        try:
            if jx_state.throttle_event.is_set():
                jx_state.throttle_event.clear()
                setattr(jx_state, "throttle_unset_ts", 0.0)
        except Exception:
            pass

    # Optional per-step hard RT budget (in milliseconds); 0 disables.
    try:
        step_ms = int(os.getenv("JINX_FRAME_STEP_RT_MS", "0"))
    except Exception:
        step_ms = 0

    async def _run_one(s: str, gid: str) -> None:
        try:
            tok = current_group.set(gid)
            t0 = time.perf_counter()
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
            try:
                _publish_event("turn.metrics", {"group": gid, "dt": time.perf_counter() - t0})
            except Exception:
                pass
        except Exception:
            # shatter() logs internally; swallow here to keep scheduler alive
            pass
        finally:
            try:
                current_group.reset(tok)  # type: ignore[name-defined]
            except Exception:
                pass

    def _score_locator_cached(msg: str) -> float:
        try:
            from jinx.micro.conversation.locator_semantics import get_locator_score_cached as _get_sc  # type: ignore
        except Exception:
            _get_sc = None  # type: ignore
        if _get_sc is None:
            return 0.0
        try:
            sc = _get_sc(msg)
        except Exception:
            sc = None
        return float(sc) if sc is not None else 0.0

    def _pop_next(dq: deque[str], prefer_simple: bool) -> str | None:
        if not dq:
            return None
        if not prefer_simple or _SIMPLE_SCAN <= 0:
            return dq.popleft()
        # Embeddings-based preference: scan first N pending for highest cached locator score
        n = min(len(dq), _SIMPLE_SCAN + 1)
        best_idx = -1
        best_sc = _LOC_THRESH
        # Enumerate without consuming: convert small head slice to list
        i = 0
        for m in dq:
            if i >= n:
                break
            sc = _score_locator_cached(m)
            if sc >= best_sc:
                best_sc = sc
                best_idx = i
            i += 1
        if best_idx <= 0:
            return dq.popleft()
        # Rotate to bring best to head, pop, then rotate back
        try:
            dq.rotate(-best_idx)
            item = dq.popleft()
            dq.rotate(best_idx)
            return item
        except Exception:
            # Fallback: pop head
            return dq.popleft()

    try:
        while True:
            # Respect global shutdown fast-path
            if jx_state.shutdown_event.is_set():
                break
            # Periodically clear throttle via TTL
            try:
                clear_throttle_if_ttl()
            except Exception:
                pass
            # Soft-throttle: only gate scheduling while there is active work; do not block intake forever
            if jx_state.throttle_event.is_set() and active:
                await asyncio.sleep(0.05)
                continue

            # Drain inbound queue briefly into per-group pending buffers
            drained = 0
            while drained < _DRAIN_MAX:  # cap per loop to avoid starvation
                try:
                    item = await asyncio.wait_for(q.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    break
                gid = _group_of(item)
                if gid not in pending_by_group:
                    pending_by_group[gid] = deque(maxlen=_GMAX)
                    groups_rr.append(gid)
                # Attempt multi-split (e.g., multiple short tasks in one message)
                subs = _split_if_multi(item, gid)
                if subs:
                    for it in subs:
                        pending_by_group[gid].append(it)
                        try:
                            _publish_event("queue.intake", {"group": gid, "text": it})
                        except Exception:
                            pass
                else:
                    pending_by_group[gid].append(item)
                drained += 1
                if not subs:
                    try:
                        _publish_event("queue.intake", {"group": gid, "text": item})
                    except Exception:
                        pass

            # Fill up to concurrency limit with fair round-robin across groups
            filled = False
            if groups_rr:
                n = len(groups_rr)
                start = rr_idx % max(1, n)
                # Two passes: prefer simple-locator, then general
                for prefer_simple in (True, False):
                    i = 0
                    while len(active) < _MAX_CONC and i < n:
                        gid = groups_rr[(start + i) % n]
                        i += 1
                        # respect per-group concurrency cap
                        if int(group_active_count.get(gid, 0)) >= _GCONC:
                            continue
                        dq = pending_by_group.get(gid)
                        if not dq:
                            continue
                        raw = _pop_next(dq, prefer_simple)
                        if raw is None:
                            continue
                        msg = _strip_group_tag(raw)
                        await _ensure_spinner()
                        t = asyncio.create_task(_run_one(msg, gid))
                        active.add(t)
                        task_group[t] = gid
                        group_active_count[gid] = int(group_active_count.get(gid, 0)) + 1
                        try:
                            _publish_event("turn.scheduled", {"group": gid, "text": msg})
                        except Exception:
                            pass
                        filled = True
                        if len(active) >= _MAX_CONC:
                            break
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
                    if gid is not None:
                        cur = int(group_active_count.get(gid, 0))
                        if cur > 0:
                            group_active_count[gid] = cur - 1
                    try:
                        _publish_event("turn.finished", {"group": gid})
                    except Exception:
                        pass
                except Exception:
                    pass

            # Prune empty groups from RR when they have no pending and are not active
            if groups_rr:
                new_rr = []
                for g in groups_rr:
                    dq = pending_by_group.get(g)
                    if (dq and len(dq) > 0) or (g in group_active_count and group_active_count[g] > 0):
                        new_rr.append(g)
                    else:
                        # drop empty group from maps
                        pending_by_group.pop(g, None)
                        group_active_count.pop(g, None)
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
