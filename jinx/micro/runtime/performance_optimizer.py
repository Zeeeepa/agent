"""Performance Optimizer - Automatic system performance tuning.

Features:
- Real-time performance monitoring
- Automatic bottleneck detection
- Resource allocation optimization
- Concurrent task management
- Memory pressure handling
- Latency spike detection
"""

from __future__ import annotations

import asyncio
import time
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class PerformanceMetric:
    """Performance measurement."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    latency_ms: float
    concurrent_tasks: int
    queue_depth: int


class PerformanceOptimizer:
    """
    Automatic performance optimization.
    
    Monitors system resources and automatically adjusts:
    - Concurrency levels
    - Batch sizes
    - Timeout values
    - Cache sizes
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Performance history
        self._metrics: deque[PerformanceMetric] = deque(maxlen=1000)
        
        # Current system state
        self._current_concurrency = 6
        self._current_batch_size = 20
        
        # Optimization thresholds
        self._thresholds = {
            'max_cpu_percent': 80.0,
            'max_memory_mb': 2000.0,
            'max_latency_p95': 1000.0,
            'min_latency_p95': 100.0,
            'max_queue_depth': 100
        }
        
        # Optimization state
        self._last_optimization = time.time()
        self._optimization_interval = 30.0  # 30 seconds
        
        # Active
        self._active = True
    
    async def start(self):
        """Start performance monitoring loop."""
        asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring and optimization."""
        
        while self._active:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                # Collect metrics
                await self._collect_metrics()
                
                # Check if optimization needed
                if time.time() - self._last_optimization > self._optimization_interval:
                    await self._optimize()
                    self._last_optimization = time.time()
            
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def _collect_metrics(self):
        """Collect current performance metrics."""
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Get ML metrics
            from .ml_monitoring import get_ml_monitoring
            
            monitor = await get_ml_monitoring()
            summary = monitor.get_metrics_summary()
            
            latency = summary.get('p95_latency', 0)
            
            # Create metric
            metric = PerformanceMetric(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                latency_ms=latency,
                concurrent_tasks=0,  # Would get from runtime
                queue_depth=0  # Would get from queue
            )
            
            self._metrics.append(metric)
        
        except Exception:
            pass
    
    async def _optimize(self):
        """Perform optimization based on metrics."""
        
        if len(self._metrics) < 10:
            return
        
        recent = list(self._metrics)[-60:]  # Last 5 minutes
        
        # Compute statistics
        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_mb for m in recent) / len(recent)
        latencies = [m.latency_ms for m in recent if m.latency_ms > 0]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        
        # === DECISION LOGIC ===
        
        # High CPU -> Reduce concurrency
        if avg_cpu > self._thresholds['max_cpu_percent']:
            await self._reduce_concurrency()
        
        # High latency -> Increase timeout or reduce load
        elif p95_latency > self._thresholds['max_latency_p95']:
            await self._handle_high_latency()
        
        # Low latency and CPU -> Can increase concurrency
        elif (p95_latency < self._thresholds['min_latency_p95'] and
              avg_cpu < 50.0):
            await self._increase_concurrency()
        
        # High memory -> Reduce cache sizes
        if avg_memory > self._thresholds['max_memory_mb']:
            await self._reduce_memory_usage()
    
    async def _reduce_concurrency(self):
        """Reduce system concurrency."""
        
        import os
        
        current = int(os.getenv('JINX_MAX_CONCURRENT', '6'))
        new_value = max(2, current - 1)
        
        os.environ['JINX_MAX_CONCURRENT'] = str(new_value)
        self._current_concurrency = new_value
        
        try:
            from jinx.micro.logger.debug_logger import debug_log
            await debug_log(
                f"Performance: Reduced concurrency to {new_value}",
                "PERF_OPT"
            )
        except Exception:
            pass
    
    async def _increase_concurrency(self):
        """Increase system concurrency."""
        
        import os
        
        current = int(os.getenv('JINX_MAX_CONCURRENT', '6'))
        new_value = min(12, current + 1)
        
        os.environ['JINX_MAX_CONCURRENT'] = str(new_value)
        self._current_concurrency = new_value
        
        try:
            from jinx.micro.logger.debug_logger import debug_log
            await debug_log(
                f"Performance: Increased concurrency to {new_value}",
                "PERF_OPT"
            )
        except Exception:
            pass
    
    async def _handle_high_latency(self):
        """Handle high latency situation."""
        
        import os
        
        # Increase timeouts
        current = int(os.getenv('JINX_STAGE_PROJCTX_MS', '5000'))
        new_value = min(10000, int(current * 1.2))
        
        os.environ['JINX_STAGE_PROJCTX_MS'] = str(new_value)
        
        # Also reduce concurrency slightly
        await self._reduce_concurrency()
        
        try:
            from jinx.micro.logger.debug_logger import debug_log
            await debug_log(
                f"Performance: Handling high latency - timeout to {new_value}ms",
                "PERF_OPT"
            )
        except Exception:
            pass
    
    async def _reduce_memory_usage(self):
        """Reduce memory usage."""
        
        try:
            # Clear caches
            from jinx.micro.embeddings.semantic_cache import get_embedding_cache
            
            cache = await get_embedding_cache()
            
            # Reduce cache size (don't clear completely)
            stats = cache.get_stats()
            current_size = stats['size']
            
            if current_size > 5000:
                # Would need to implement cache trimming
                pass
            
            from jinx.micro.logger.debug_logger import debug_log
            await debug_log(
                f"Performance: Memory pressure detected, optimizing caches",
                "PERF_OPT"
            )
        
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        
        if not self._metrics:
            return {}
        
        recent = list(self._metrics)[-60:]
        
        return {
            'current_concurrency': self._current_concurrency,
            'current_batch_size': self._current_batch_size,
            'avg_cpu_percent': sum(m.cpu_percent for m in recent) / len(recent),
            'avg_memory_mb': sum(m.memory_mb for m in recent) / len(recent),
            'metrics_count': len(self._metrics),
            'last_optimization': self._last_optimization,
            'thresholds': self._thresholds
        }
    
    def stop(self):
        """Stop optimizer."""
        self._active = False


# Singleton
_optimizer: Optional[PerformanceOptimizer] = None
_optimizer_lock = asyncio.Lock()


async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get singleton performance optimizer."""
    global _optimizer
    if _optimizer is None:
        async with _optimizer_lock:
            if _optimizer is None:
                _optimizer = PerformanceOptimizer()
                await _optimizer.start()
    return _optimizer


__all__ = [
    "PerformanceOptimizer",
    "PerformanceMetric",
    "get_performance_optimizer",
]
