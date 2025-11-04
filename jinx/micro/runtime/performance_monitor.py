"""Performance Monitor - real-time monitoring with adaptive learning.

Automatically monitors and learns:
- Operation latencies
- Success rates
- Resource usage
- Workload patterns
- Optimal configurations
"""

from __future__ import annotations

import asyncio
import time
import functools
from typing import Any, Callable, Optional, TypeVar, cast

T = TypeVar('T')


def monitor_performance(
    workload_type: str = 'default',
    record_to_adaptive_config: bool = True
):
    """Decorator to monitor function performance."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            success = False
            result = None
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            
            except Exception as e:
                success = False
                raise
            
            finally:
                latency_ms = (time.time() - start_time) * 1000
                
                # Record to adaptive config
                if record_to_adaptive_config:
                    try:
                        from .adaptive_config import record_operation_result
                        asyncio.create_task(
                            record_operation_result(latency_ms, success, workload_type)
                        )
                    except Exception:
                        pass
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # For sync functions, wrap in simple timing
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                
                # Record async
                if record_to_adaptive_config:
                    try:
                        from .adaptive_config import record_operation_result
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(
                                record_operation_result(latency_ms, success, workload_type)
                            )
                    except Exception:
                        pass
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[..., T], async_wrapper)
        else:
            return cast(Callable[..., T], sync_wrapper)
    
    return decorator


class PerformanceContext:
    """Context manager for monitoring code blocks."""
    
    def __init__(self, operation_name: str, workload_type: str = 'default'):
        self.operation_name = operation_name
        self.workload_type = workload_type
        self.start_time: Optional[float] = None
        self.success = False
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency_ms = (time.time() - self.start_time) * 1000
            self.success = exc_type is None
            
            # Record metrics
            try:
                from .adaptive_config import record_operation_result
                await record_operation_result(
                    latency_ms, self.success, self.workload_type
                )
            except Exception:
                pass
        
        return False  # Don't suppress exceptions


class RealTimeOptimizer:
    """Real-time configuration optimizer."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._optimization_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start real-time optimization."""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._optimization_task = asyncio.create_task(self._optimization_loop())
    
    async def stop(self):
        """Stop optimization."""
        async with self._lock:
            self._running = False
            
            if self._optimization_task:
                self._optimization_task.cancel()
                try:
                    await self._optimization_task
                except asyncio.CancelledError:
                    pass
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        
        while self._running:
            try:
                # Get adaptive config
                from .adaptive_config import get_adaptive_config
                
                config = await get_adaptive_config()
                metrics = config.get_metrics()
                
                # Check performance
                perf = metrics['performance']
                
                # Log metrics to brain memory
                if perf['avg_latency_ms'] > 0:
                    try:
                        from jinx.micro.brain import remember_episode
                        
                        await remember_episode(
                            content=f"Performance: latency={perf['avg_latency_ms']:.1f}ms, "
                                   f"success={perf['success_rate']:.2%}, "
                                   f"p95={perf['p95_latency_ms']:.1f}ms",
                            episode_type='performance',
                            context={
                                'metrics': perf,
                                'workload': metrics['workload']
                            },
                            importance=0.5
                        )
                    except Exception:
                        pass
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(30)


# Global optimizer
_optimizer: Optional[RealTimeOptimizer] = None
_optimizer_lock = asyncio.Lock()


async def get_optimizer() -> RealTimeOptimizer:
    """Get singleton optimizer."""
    global _optimizer
    if _optimizer is None:
        async with _optimizer_lock:
            if _optimizer is None:
                _optimizer = RealTimeOptimizer()
    return _optimizer


async def start_real_time_optimization():
    """Start real-time optimization."""
    optimizer = await get_optimizer()
    await optimizer.start()


async def stop_real_time_optimization():
    """Stop real-time optimization."""
    optimizer = await get_optimizer()
    await optimizer.stop()


__all__ = [
    "monitor_performance",
    "PerformanceContext",
    "RealTimeOptimizer",
    "get_optimizer",
    "start_real_time_optimization",
    "stop_real_time_optimization",
]
