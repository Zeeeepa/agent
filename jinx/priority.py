from __future__ import annotations

import asyncio
import heapq
import contextlib
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict

from jinx.settings import Settings


@dataclass
class PriorityMetrics:
    """Real-time metrics for priority queue performance."""
    total_processed: int = 0
    high_priority_count: int = 0
    normal_priority_count: int = 0
    low_priority_count: int = 0
    avg_queue_time_ms: float = 0.0
    max_queue_time_ms: float = 0.0


def classify_priority(msg: str) -> int:
    """
    Advanced message priority classification with multi-level detection.

    Priority levels (lower = higher priority):
    - 0: Critical/Urgent (!, /urgent, ASAP)
    - 1: High (!!, /high, /important)
    - 2: Normal (default)
    - 3: Low (/low, /defer)
    - 4: Background (e.g., '<no_response>', /background)
    """
    s = msg.strip().lower()
    if not s:
        return 2
    
    # Level 0: Critical/Urgent
    if any(s.startswith(prefix) for prefix in ['!!', '/urgent', '/critical']):
        return 0
    if 'asap' in s[:20] or 'emergency' in s[:30]:
        return 0
    
    # Level 1: High priority
    if s.startswith('!') and not s.startswith('!!'):
        return 1
    if any(s.startswith(prefix) for prefix in ['/high', '/important', '/priority']):
        return 1
    
    # Level 3: Low priority
    if any(s.startswith(prefix) for prefix in ['/low', '/defer', '/later']):
        return 3
    
    # Level 4: Background
    if s == '<no_response>' or s.startswith('/background'):
        return 4
    
    # Level 2: Normal (default)
    return 2


def start_priority_dispatcher_task(src: "asyncio.Queue[str]", dst: "asyncio.Queue[str]", settings: Settings) -> "asyncio.Task[None]":
    async def _run() -> None:
        # Advanced priority dispatcher with metrics and starvation prevention
        loop = asyncio.get_running_loop()
        budget = max(1, settings.runtime.hard_rt_budget_ms) / 1000.0
        next_yield = loop.time() + budget

        heap: List[Tuple[int, int, float, str]] = []  # (priority, seq, enqueue_time, msg)
        seq = 0
        metrics = PriorityMetrics()
        
        # Starvation prevention: track time since last low-priority dispatch
        last_low_priority_dispatch: Dict[int, float] = defaultdict(lambda: loop.time())
        starvation_threshold_s = 30.0  # Boost priority after 30s wait

        async def _awaitable_src_get() -> str:
            return await src.get()

        while True:
            if not settings.runtime.use_priority_queue:
                # Fast path: FIFO pass-through with cooperative yield
                msg = await src.get()
                await dst.put(msg)
                metrics.total_processed += 1
                if loop.time() >= next_yield:
                    await asyncio.sleep(0)
                    next_yield = loop.time() + budget
                continue

            # Priority mode: race new item vs flush tick
            get_task = asyncio.create_task(_awaitable_src_get())
            flush_task = asyncio.create_task(asyncio.sleep(0.001))  # Small delay for batching
            try:
                done, _ = await asyncio.wait({get_task, flush_task}, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                get_task.cancel(); flush_task.cancel()
                raise

            if get_task in done:
                try:
                    msg = get_task.result()
                except asyncio.CancelledError:
                    raise
                pr = classify_priority(msg)
                enqueue_time = loop.time()
                heapq.heappush(heap, (pr, seq, enqueue_time, msg))
                seq += 1
                
                # Update metrics
                if pr == 0:
                    metrics.high_priority_count += 1
                elif pr == 1:
                    metrics.normal_priority_count += 1
                else:
                    metrics.low_priority_count += 1
                    
            # Cancel pending tasks
            if not get_task.done():
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
            if not flush_task.done():
                flush_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await flush_task

            if heap:
                # Anti-starvation: check if any low-priority item is starving
                current_time = loop.time()
                
                # Rebuild heap with boosted priorities for starving items
                modified = False
                for i, (pr, s, enq_t, m) in enumerate(heap):
                    if pr >= 2:  # Low priority
                        wait_time = current_time - enq_t
                        if wait_time >= starvation_threshold_s:
                            # Boost priority to prevent starvation
                            heap[i] = (max(0, pr - 2), s, enq_t, m)
                            modified = True
                
                if modified:
                    heapq.heapify(heap)
                
                # Dispatch highest priority item
                pr, s, enq_t, item = heapq.heappop(heap)
                await dst.put(item)
                
                # Update metrics
                queue_time_ms = (current_time - enq_t) * 1000.0
                metrics.total_processed += 1
                metrics.avg_queue_time_ms = (
                    (metrics.avg_queue_time_ms * (metrics.total_processed - 1) + queue_time_ms)
                    / metrics.total_processed
                )
                metrics.max_queue_time_ms = max(metrics.max_queue_time_ms, queue_time_ms)
                last_low_priority_dispatch[pr] = current_time

            if loop.time() >= next_yield:
                await asyncio.sleep(0)
                next_yield = loop.time() + budget
                
                # Periodically log metrics (every ~1000 messages)
                if metrics.total_processed % 1000 == 0 and metrics.total_processed > 0:
                    try:
                        from jinx.logging_service import bomb_log
                        await bomb_log(
                            f"Priority queue metrics: processed={metrics.total_processed}, "
                            f"avg_wait={metrics.avg_queue_time_ms:.1f}ms, "
                            f"max_wait={metrics.max_queue_time_ms:.1f}ms"
                        )
                    except Exception:
                        pass

    return asyncio.create_task(_run(), name="priority-dispatcher-service")
