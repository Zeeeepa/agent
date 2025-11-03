"""Advanced hot-store cache with metrics and memory management."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import contextlib
import threading


@dataclass
class HotStoreMetrics:
    """Metrics for hot store cache performance."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    background_refreshes: int = 0
    refresh_errors: int = 0
    last_refresh_time_ms: float = 0.0
    snapshot_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total)


@dataclass
class SnapshotData:
    """Enhanced snapshot with metadata."""
    data: List[Any]
    timestamp_ms: float
    generation: int = 0
    size_bytes: int = 0


# Thread-safe global state
_SNAPSHOTS: Dict[str, SnapshotData] = {}
_TASKS: Dict[str, asyncio.Task] = {}
_LOCKS: Dict[str, asyncio.Lock] = {}
_METRICS: Dict[str, HotStoreMetrics] = {}
_GLOBAL_LOCK = threading.RLock()


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


async def get_hot_snapshot(key: str, loader: Callable[[], Awaitable[List[Any]]], ttl_ms: int) -> List[Any]:
    """Get cached snapshot with automatic background refresh.
    
    Features:
    - TTL-based automatic refresh
    - Non-blocking background updates
    - Metrics tracking
    - Memory-efficient storage
    - Thread-safe access
    """
    ttl_ms = max(0, int(ttl_ms))
    
    # Initialize metrics for this key if needed
    with _GLOBAL_LOCK:
        if key not in _METRICS:
            _METRICS[key] = HotStoreMetrics()
        metrics = _METRICS[key]
        metrics.total_requests += 1
    
    # Get current snapshot
    snap_data = _SNAPSHOTS.get(key)
    current_time = _now_ms()
    
    # Serve fresh snapshot if within TTL
    if snap_data and ttl_ms > 0 and (current_time - snap_data.timestamp_ms) <= ttl_ms:
        with _GLOBAL_LOCK:
            metrics.cache_hits += 1
        return snap_data.data
    
    # If a refresh is running, serve current snapshot without blocking
    t = _TASKS.get(key)
    if t is not None and not t.done():
        return snap_data.data if snap_data else []
    
    # Lock per key to avoid double refresh
    lock = _LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        # Double-check after acquiring lock
        snap_data = _SNAPSHOTS.get(key)
        if snap_data and ttl_ms > 0 and (current_time - snap_data.timestamp_ms) <= ttl_ms:
            with _GLOBAL_LOCK:
                metrics.cache_hits += 1
            return snap_data.data
        
        with _GLOBAL_LOCK:
            metrics.cache_misses += 1
        
        # Start loader task with timing
        load_start = time.perf_counter()
        task = asyncio.create_task(loader())
        _TASKS[key] = task
        
        # If we have no snapshot yet, await first load; else refresh in background
        if not snap_data or not snap_data.data:
            try:
                res = await task
                elapsed_ms = (time.perf_counter() - load_start) * 1000.0
                
                # Calculate approximate size
                import sys
                size_bytes = sys.getsizeof(res)
                
                new_snap = SnapshotData(
                    data=res or [],
                    timestamp_ms=_now_ms(),
                    generation=0,
                    size_bytes=size_bytes
                )
                _SNAPSHOTS[key] = new_snap
                
                with _GLOBAL_LOCK:
                    metrics.last_refresh_time_ms = elapsed_ms
                    metrics.snapshot_size = len(res or [])
            except BaseException as e:
                # On cancellation or any failure provide empty result
                with _GLOBAL_LOCK:
                    metrics.refresh_errors += 1
                res = []
                _SNAPSHOTS[key] = SnapshotData(data=[], timestamp_ms=_now_ms())
            finally:
                # Clear task reference once finished
                _TASKS.pop(key, None)
            
            return _SNAPSHOTS[key].data

        # Background refresh with callback
        def _on_done(t: asyncio.Task) -> None:
            elapsed_ms = (time.perf_counter() - load_start) * 1000.0
            
            try:
                # If task was cancelled, just drop it silently
                if t.cancelled():
                    with _GLOBAL_LOCK:
                        metrics.refresh_errors += 1
                    return
                
                res = t.result()
                import sys
                size_bytes = sys.getsizeof(res)
                
                old_snap = _SNAPSHOTS.get(key)
                new_generation = (old_snap.generation + 1) if old_snap else 0
                
                new_snap = SnapshotData(
                    data=res or [],
                    timestamp_ms=_now_ms(),
                    generation=new_generation,
                    size_bytes=size_bytes
                )
                _SNAPSHOTS[key] = new_snap
                
                with _GLOBAL_LOCK:
                    metrics.background_refreshes += 1
                    metrics.last_refresh_time_ms = elapsed_ms
                    metrics.snapshot_size = len(res or [])
            except BaseException:
                # Swallow any exception from background refresh
                with _GLOBAL_LOCK:
                    metrics.refresh_errors += 1
                return
            finally:
                # Ensure we clear the task reference regardless of outcome
                with contextlib.suppress(Exception):
                    _TASKS.pop(key, None)

        task.add_done_callback(_on_done)
        return snap_data.data if snap_data else []


# Convenience wrappers
async def get_runtime_items_hot(loader: Callable[[], Awaitable[List[Any]]], ttl_ms: int) -> List[Any]:
    """Get runtime items with hot caching."""
    return await get_hot_snapshot("runtime_items", loader, ttl_ms)


async def get_project_chunks_hot(loader: Callable[[], Awaitable[List[Any]]], ttl_ms: int) -> List[Any]:
    """Get project chunks with hot caching."""
    return await get_hot_snapshot("project_chunks", loader, ttl_ms)


def get_metrics(key: str) -> Optional[HotStoreMetrics]:
    """Get metrics for a specific cache key."""
    with _GLOBAL_LOCK:
        return _METRICS.get(key)


def get_all_metrics() -> Dict[str, HotStoreMetrics]:
    """Get metrics for all cache keys."""
    with _GLOBAL_LOCK:
        return dict(_METRICS)


def clear_cache(key: Optional[str] = None) -> None:
    """Clear cache for a specific key or all caches."""
    with _GLOBAL_LOCK:
        if key:
            _SNAPSHOTS.pop(key, None)
            _TASKS.pop(key, None)
            _LOCKS.pop(key, None)
            _METRICS.pop(key, None)
        else:
            _SNAPSHOTS.clear()
            _TASKS.clear()
            _LOCKS.clear()
            _METRICS.clear()
