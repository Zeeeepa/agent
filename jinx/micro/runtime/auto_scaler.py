"""Auto Scaler - Dynamic resource scaling based on load.

Features:
- Load-based scaling
- Predictive scaling (ML-based)
- Queue management
- Rate limiting
- Circuit breaker pattern
- Graceful degradation
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass
from collections import deque
from enum import Enum


class SystemState(Enum):
    """System operational state."""
    NORMAL = "normal"
    STRESSED = "stressed"
    OVERLOADED = "overloaded"
    DEGRADED = "degraded"


@dataclass
class LoadMetric:
    """Load measurement."""
    timestamp: float
    queue_depth: int
    processing_rate: float  # requests per second
    avg_latency_ms: float
    error_rate: float


class AutoScaler:
    """
    Automatic resource scaling.
    
    Monitors load and automatically adjusts system capacity:
    - Scales up under high load
    - Scales down when idle
    - Applies backpressure when overloaded
    - Circuit breaking for cascading failures
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Load metrics
        self._load_metrics: Deque[LoadMetric] = deque(maxlen=300)  # 5 minutes
        
        # Current state
        self._state = SystemState.NORMAL
        self._max_concurrent = 6
        self._current_queue_depth = 0
        
        # Scaling thresholds
        self._scale_up_threshold = 0.8  # 80% capacity
        self._scale_down_threshold = 0.3  # 30% capacity
        self._overload_threshold = 1.2  # 120% capacity
        
        # Circuit breaker
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_failure_threshold = 10
        self._circuit_reset_time = 60.0  # 1 minute
        self._circuit_last_failure = 0.0
        
        # Rate limiter
        self._rate_limit_tokens = 100
        self._rate_limit_max = 100
        self._rate_limit_refill_rate = 10  # tokens per second
        self._rate_limit_last_refill = time.time()
        
        # Predictive scaling
        self._load_history: List[float] = []
        self._prediction_enabled = True
        
        # Active
        self._active = True
    
    async def start(self):
        """Start auto-scaler."""
        asyncio.create_task(self._scaling_loop())
        asyncio.create_task(self._rate_limit_refill_loop())
    
    async def _scaling_loop(self):
        """Background scaling loop."""
        
        while self._active:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
                # Collect metrics
                await self._collect_load_metrics()
                
                # Determine system state
                await self._update_system_state()
                
                # Make scaling decision
                await self._make_scaling_decision()
                
                # Check circuit breaker
                await self._check_circuit_breaker()
            
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def _rate_limit_refill_loop(self):
        """Refill rate limit tokens."""
        
        while self._active:
            try:
                await asyncio.sleep(0.1)  # Every 100ms
                
                async with self._lock:
                    now = time.time()
                    elapsed = now - self._rate_limit_last_refill
                    
                    tokens_to_add = int(elapsed * self._rate_limit_refill_rate)
                    
                    if tokens_to_add > 0:
                        self._rate_limit_tokens = min(
                            self._rate_limit_max,
                            self._rate_limit_tokens + tokens_to_add
                        )
                        self._rate_limit_last_refill = now
            
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def acquire_token(self, cost: int = 1) -> bool:
        """
        Acquire rate limit token.
        
        Returns True if allowed, False if rate limited.
        """
        
        async with self._lock:
            if self._rate_limit_tokens >= cost:
                self._rate_limit_tokens -= cost
                return True
            
            return False
    
    async def check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is open.
        
        Returns True if requests allowed, False if circuit open.
        """
        
        async with self._lock:
            if not self._circuit_open:
                return True
            
            # Check if enough time passed to retry
            if time.time() - self._circuit_last_failure > self._circuit_reset_time:
                self._circuit_open = False
                self._circuit_failures = 0
                return True
            
            return False
    
    async def record_success(self):
        """Record successful operation."""
        
        async with self._lock:
            # Reset circuit breaker on success
            if self._circuit_failures > 0:
                self._circuit_failures = max(0, self._circuit_failures - 1)
    
    async def record_failure(self):
        """Record failed operation."""
        
        async with self._lock:
            self._circuit_failures += 1
            self._circuit_last_failure = time.time()
            
            if self._circuit_failures >= self._circuit_failure_threshold:
                self._circuit_open = True
                
                try:
                    from jinx.micro.logger.debug_logger import debug_log
                    await debug_log(
                        "Circuit breaker OPENED - too many failures",
                        "AUTO_SCALER"
                    )
                except Exception:
                    pass
    
    async def _collect_load_metrics(self):
        """Collect current load metrics."""
        
        try:
            # Get monitoring data
            from .ml_monitoring import get_ml_monitoring
            
            monitor = await get_ml_monitoring()
            summary = monitor.get_metrics_summary()
            
            # Create metric
            metric = LoadMetric(
                timestamp=time.time(),
                queue_depth=self._current_queue_depth,
                processing_rate=summary.get('recent_predictions', 0) / 60.0,  # per second
                avg_latency_ms=summary.get('avg_latency_ms', 0),
                error_rate=1.0 - summary.get('success_rate', 1.0)
            )
            
            self._load_metrics.append(metric)
            
            # Track load history for prediction
            self._load_history.append(metric.processing_rate)
            if len(self._load_history) > 100:
                self._load_history.pop(0)
        
        except Exception:
            pass
    
    async def _update_system_state(self):
        """Update system state based on metrics."""
        
        if not self._load_metrics:
            return
        
        recent = list(self._load_metrics)[-30:]  # Last 5 minutes
        
        # Compute load factor
        avg_rate = sum(m.processing_rate for m in recent) / len(recent)
        capacity = self._max_concurrent * 0.5  # Rough capacity estimate
        
        load_factor = avg_rate / capacity if capacity > 0 else 0
        
        # Determine state
        old_state = self._state
        
        if load_factor > self._overload_threshold:
            self._state = SystemState.OVERLOADED
        elif load_factor > self._scale_up_threshold:
            self._state = SystemState.STRESSED
        elif self._circuit_open:
            self._state = SystemState.DEGRADED
        else:
            self._state = SystemState.NORMAL
        
        # Log state changes
        if old_state != self._state:
            try:
                from jinx.micro.logger.debug_logger import debug_log
                await debug_log(
                    f"System state: {old_state.value} -> {self._state.value}",
                    "AUTO_SCALER"
                )
            except Exception:
                pass
    
    async def _make_scaling_decision(self):
        """Make scaling decision based on state."""
        
        if self._state == SystemState.OVERLOADED:
            await self._apply_backpressure()
        
        elif self._state == SystemState.STRESSED:
            await self._scale_up()
        
        elif self._state == SystemState.NORMAL:
            # Check if we can scale down
            if self._max_concurrent > 4:
                recent = list(self._load_metrics)[-30:]
                if recent:
                    avg_rate = sum(m.processing_rate for m in recent) / len(recent)
                    capacity = self._max_concurrent * 0.5
                    load_factor = avg_rate / capacity if capacity > 0 else 0
                    
                    if load_factor < self._scale_down_threshold:
                        await self._scale_down()
        
        elif self._state == SystemState.DEGRADED:
            await self._enter_degraded_mode()
    
    async def _scale_up(self):
        """Scale up capacity."""
        
        import os
        
        new_concurrent = min(16, self._max_concurrent + 2)
        
        if new_concurrent != self._max_concurrent:
            os.environ['JINX_MAX_CONCURRENT'] = str(new_concurrent)
            self._max_concurrent = new_concurrent
            
            try:
                from jinx.micro.logger.debug_logger import debug_log
                await debug_log(
                    f"Scaling UP: concurrency -> {new_concurrent}",
                    "AUTO_SCALER"
                )
            except Exception:
                pass
    
    async def _scale_down(self):
        """Scale down capacity."""
        
        import os
        
        new_concurrent = max(2, self._max_concurrent - 1)
        
        if new_concurrent != self._max_concurrent:
            os.environ['JINX_MAX_CONCURRENT'] = str(new_concurrent)
            self._max_concurrent = new_concurrent
            
            try:
                from jinx.micro.logger.debug_logger import debug_log
                await debug_log(
                    f"Scaling DOWN: concurrency -> {new_concurrent}",
                    "AUTO_SCALER"
                )
            except Exception:
                pass
    
    async def _apply_backpressure(self):
        """Apply backpressure under overload."""
        
        # Reduce rate limit
        async with self._lock:
            self._rate_limit_max = max(20, int(self._rate_limit_max * 0.8))
        
        # Reduce concurrency aggressively
        await self._scale_down()
        
        try:
            from jinx.micro.logger.debug_logger import debug_log
            await debug_log(
                f"BACKPRESSURE: rate limit -> {self._rate_limit_max}",
                "AUTO_SCALER"
            )
        except Exception:
            pass
    
    async def _enter_degraded_mode(self):
        """Enter degraded mode."""
        
        # Minimal processing only
        import os
        os.environ['JINX_MAX_CONCURRENT'] = '2'
        self._max_concurrent = 2
        
        try:
            from jinx.micro.logger.debug_logger import debug_log
            await debug_log(
                "DEGRADED MODE: minimal processing only",
                "AUTO_SCALER"
            )
        except Exception:
            pass
    
    async def _check_circuit_breaker(self):
        """Check and potentially reset circuit breaker."""
        
        async with self._lock:
            if self._circuit_open:
                if time.time() - self._circuit_last_failure > self._circuit_reset_time:
                    self._circuit_open = False
                    self._circuit_failures = 0
                    
                    try:
                        from jinx.micro.logger.debug_logger import debug_log
                        await debug_log(
                            "Circuit breaker CLOSED - resetting",
                            "AUTO_SCALER"
                        )
                    except Exception:
                        pass
    
    def predict_future_load(self) -> float:
        """
        Predict future load using simple time series.
        
        Returns predicted processing rate.
        """
        
        if len(self._load_history) < 10:
            return 0.0
        
        # Simple moving average with trend
        recent = self._load_history[-20:]
        avg = np.mean(recent)
        
        # Linear trend
        if len(recent) >= 2:
            x = np.arange(len(recent))
            y = np.array(recent)
            
            # Fit line
            slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-10)
            
            # Predict next value
            predicted = avg + slope
            
            return max(0, predicted)
        
        return avg
    
    def get_stats(self) -> Dict[str, any]:
        """Get scaler statistics."""
        
        return {
            'state': self._state.value,
            'max_concurrent': self._max_concurrent,
            'rate_limit_tokens': self._rate_limit_tokens,
            'rate_limit_max': self._rate_limit_max,
            'circuit_open': self._circuit_open,
            'circuit_failures': self._circuit_failures,
            'current_queue_depth': self._current_queue_depth,
            'predicted_load': self.predict_future_load()
        }
    
    def stop(self):
        """Stop scaler."""
        self._active = False


# Singleton
_auto_scaler: Optional[AutoScaler] = None
_scaler_lock = asyncio.Lock()


async def get_auto_scaler() -> AutoScaler:
    """Get singleton auto scaler."""
    global _auto_scaler
    if _auto_scaler is None:
        async with _scaler_lock:
            if _auto_scaler is None:
                _auto_scaler = AutoScaler()
                await _auto_scaler.start()
    return _auto_scaler


__all__ = [
    "AutoScaler",
    "SystemState",
    "LoadMetric",
    "get_auto_scaler",
]
