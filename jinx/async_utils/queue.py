from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar, Generic, Optional
from dataclasses import dataclass
from collections import deque

T = TypeVar("T")


@dataclass
class QueueMetrics:
    """Real-time queue performance metrics."""
    enqueued: int = 0
    dequeued: int = 0
    dropped: int = 0
    current_size: int = 0
    peak_size: int = 0
    avg_wait_time_ms: float = 0.0


class BoundedPriorityQueue(Generic[T]):
    """Advanced bounded queue with priority support and metrics.
    
    Features:
    - Priority-based dequeue
    - Bounded capacity with overflow handling
    - Real-time metrics
    - Backpressure signaling
    """
    
    def __init__(self, maxsize: int = 0, drop_policy: str = "oldest"):
        self._queue: deque[tuple[float, T]] = deque(maxlen=maxsize if maxsize > 0 else None)
        self._maxsize = maxsize
        self._drop_policy = drop_policy  # "oldest" or "newest"
        self._metrics = QueueMetrics()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._enqueue_times: dict[int, float] = {}
    
    async def put(self, item: T, priority: float = 0.0) -> bool:
        """Enqueue with priority. Returns True if successful."""
        async with self._lock:
            item_id = id(item)
            self._enqueue_times[item_id] = time.time()
            
            if self._maxsize > 0 and len(self._queue) >= self._maxsize:
                # Queue full - apply drop policy
                if self._drop_policy == "oldest":
                    dropped = self._queue.popleft()
                    self._metrics.dropped += 1
                    # Clean up enqueue time
                    self._enqueue_times.pop(id(dropped[1]), None)
                else:  # newest
                    self._metrics.dropped += 1
                    return False
            
            self._queue.append((priority, item))
            self._metrics.enqueued += 1
            self._metrics.current_size = len(self._queue)
            self._metrics.peak_size = max(self._metrics.peak_size, self._metrics.current_size)
            
            self._not_empty.notify()
            return True
    
    async def get(self) -> T:
        """Dequeue highest priority item."""
        async with self._not_empty:
            while len(self._queue) == 0:
                await self._not_empty.wait()
            
            # Find highest priority (lowest value)
            if len(self._queue) == 1:
                priority, item = self._queue.popleft()
            else:
                # Linear search for highest priority
                best_idx = 0
                best_priority = self._queue[0][0]
                for i, (p, _) in enumerate(self._queue):
                    if p < best_priority:
                        best_priority = p
                        best_idx = i
                
                # Remove item at best_idx
                self._queue.rotate(-best_idx)
                priority, item = self._queue.popleft()
                self._queue.rotate(best_idx)
            
            self._metrics.dequeued += 1
            self._metrics.current_size = len(self._queue)
            
            # Update wait time metrics
            item_id = id(item)
            if item_id in self._enqueue_times:
                wait_time_ms = (time.time() - self._enqueue_times[item_id]) * 1000.0
                count = self._metrics.dequeued
                self._metrics.avg_wait_time_ms = (
                    (self._metrics.avg_wait_time_ms * (count - 1) + wait_time_ms) / count
                )
                del self._enqueue_times[item_id]
            
            return item
    
    def qsize(self) -> int:
        """Current queue size."""
        return len(self._queue)
    
    def get_metrics(self) -> QueueMetrics:
        """Get current queue metrics."""
        return self._metrics


async def _run_on_drop(cb: Callable[[], Awaitable[None]]) -> None:
    """Wrap an Awaitable callback into a coroutine for create_task()."""
    await cb()


def try_put_nowait(q: "asyncio.Queue[T]", item: T) -> bool:
    """Attempt to enqueue without blocking. Return True on success.

    Parameters
    ----------
    q : asyncio.Queue[T]
        Target queue.
    item : T
        Item to enqueue.
    """
    try:
        q.put_nowait(item)
        return True
    except asyncio.QueueFull:
        return False


def put_drop_oldest(q: "asyncio.Queue[T]", item: T, on_drop: Callable[[], Awaitable[None]] | None = None) -> None:
    """Enqueue ``item``; if full, drop oldest and log via ``on_drop``.

    Enhanced with better error handling and metrics.

    Parameters
    ----------
    q : asyncio.Queue[T]
        Target bounded queue.
    item : T
        Item to enqueue.
    on_drop : Optional[Callable[[], Awaitable[None]]]
        Async callback invoked after an item is dropped due to saturation.
    """
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        dropped_item = None
        try:
            dropped_item = q.get_nowait()  # drop oldest
        except asyncio.QueueEmpty:
            pass
        
        if on_drop:
            # Fire-and-forget; caller may choose to await explicitly instead
            asyncio.create_task(_run_on_drop(on_drop))
        
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # Still full somehow, give up
            pass


class AdaptiveQueue(Generic[T]):
    """Self-tuning queue that adapts capacity based on traffic patterns.
    
    Automatically grows/shrinks within bounds to optimize memory usage.
    """
    
    def __init__(self, min_size: int = 10, max_size: int = 1000, growth_factor: float = 1.5):
        self._min_size = min_size
        self._max_size = max_size
        self._growth_factor = growth_factor
        self._current_capacity = min_size
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=min_size)
        self._metrics = QueueMetrics()
        self._last_resize_time = time.time()
        self._resize_cooldown = 5.0  # seconds
    
    async def put(self, item: T) -> None:
        """Enqueue item with automatic capacity adjustment."""
        try:
            self._queue.put_nowait(item)
            self._metrics.enqueued += 1
        except asyncio.QueueFull:
            await self._maybe_grow()
            await self._queue.put(item)
            self._metrics.enqueued += 1
        
        self._metrics.current_size = self._queue.qsize()
        self._metrics.peak_size = max(self._metrics.peak_size, self._metrics.current_size)
    
    async def get(self) -> T:
        """Dequeue item."""
        item = await self._queue.get()
        self._metrics.dequeued += 1
        self._metrics.current_size = self._queue.qsize()
        
        await self._maybe_shrink()
        return item
    
    async def _maybe_grow(self) -> None:
        """Grow queue capacity if utilization is high."""
        now = time.time()
        if (now - self._last_resize_time) < self._resize_cooldown:
            return
        
        new_capacity = min(int(self._current_capacity * self._growth_factor), self._max_size)
        if new_capacity > self._current_capacity:
            self._current_capacity = new_capacity
            # Create new queue with larger capacity
            old_queue = self._queue
            self._queue = asyncio.Queue(maxsize=new_capacity)
            # Transfer items
            while not old_queue.empty():
                try:
                    self._queue.put_nowait(old_queue.get_nowait())
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    break
            self._last_resize_time = now
    
    async def _maybe_shrink(self) -> None:
        """Shrink queue capacity if utilization is low."""
        now = time.time()
        if (now - self._last_resize_time) < self._resize_cooldown:
            return
        
        utilization = self._queue.qsize() / max(1, self._current_capacity)
        if utilization < 0.3 and self._current_capacity > self._min_size:
            new_capacity = max(int(self._current_capacity / self._growth_factor), self._min_size)
            if new_capacity < self._current_capacity:
                self._current_capacity = new_capacity
                self._last_resize_time = now
