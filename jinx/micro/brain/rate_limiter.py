"""Self-tuning adaptive rate limiter with congestion detection.

Replaces fixed rate limits with dynamic, load-aware throttling.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class RateLimitState:
    """Current rate limit state."""
    current_limit: float  # requests per second
    window_start: float
    request_count: int
    denied_count: int
    avg_latency: float


class AdaptiveRateLimiter:
    """Self-tuning rate limiter that adapts to system load."""
    
    def __init__(
        self,
        initial_limit: float = 10.0,
        min_limit: float = 1.0,
        max_limit: float = 100.0,
        window_seconds: float = 1.0,
        state_path: str = "log/rate_limiter.json"
    ):
        self.current_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.window_seconds = window_seconds
        self.state_path = state_path
        
        # Sliding window for request timestamps
        self.requests: Deque[float] = deque(maxlen=1000)
        
        # Latency tracking for congestion detection
        self.latencies: Deque[float] = deque(maxlen=100)
        
        # Adaptation parameters
        self.increase_factor = 1.1
        self.decrease_factor = 0.9
        self.adaptation_interval = 5.0  # seconds
        self.last_adaptation = time.time()
        
        # Statistics
        self.total_requests = 0
        self.total_denied = 0
        self.total_allowed = 0
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.current_limit = data.get('current_limit', self.current_limit)
                self.total_requests = data.get('total_requests', 0)
                self.total_denied = data.get('total_denied', 0)
                self.total_allowed = data.get('total_allowed', 0)
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                data = {
                    'current_limit': self.current_limit,
                    'total_requests': self.total_requests,
                    'total_denied': self.total_denied,
                    'total_allowed': self.total_allowed,
                    'timestamp': time.time()
                }
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _clean_old_requests(self, now: float) -> None:
        """Remove requests outside the current window."""
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    def _detect_congestion(self) -> bool:
        """Detect system congestion from latency trends."""
        if len(self.latencies) < 10:
            return False
        
        # Recent latency vs historical
        recent = list(self.latencies)[-10:]
        historical = list(self.latencies)[:-10] if len(self.latencies) > 10 else []
        
        if not historical:
            return False
        
        recent_avg = sum(recent) / len(recent)
        historical_avg = sum(historical) / len(historical)
        
        # Congestion if recent latency significantly higher
        return recent_avg > historical_avg * 1.5
    
    def _should_adapt(self, now: float) -> bool:
        """Check if it's time to adapt the limit."""
        return (now - self.last_adaptation) >= self.adaptation_interval
    
    async def _adapt_limit(self, now: float) -> None:
        """Adapt rate limit based on observed behavior."""
        if not self._should_adapt(now):
            return
        
        self.last_adaptation = now
        
        # Calculate current rate
        window_requests = sum(1 for t in self.requests if t >= now - self.window_seconds)
        current_rate = window_requests / self.window_seconds
        
        # Detect congestion
        congested = self._detect_congestion()
        
        # Utilization
        utilization = current_rate / max(1.0, self.current_limit)
        
        # Adaptation logic
        if congested:
            # Reduce limit due to congestion
            new_limit = max(self.min_limit, self.current_limit * self.decrease_factor)
        elif utilization > 0.9:
            # High utilization but not congested -> can increase
            new_limit = min(self.max_limit, self.current_limit * self.increase_factor)
        elif utilization < 0.3:
            # Low utilization -> can afford to increase
            new_limit = min(self.max_limit, self.current_limit * self.increase_factor)
        else:
            # Stable region -> no change
            new_limit = self.current_limit
        
        if new_limit != self.current_limit:
            self.current_limit = new_limit
            
            # Log adaptation
            try:
                if os.getenv("JINX_RATE_LIMITER_LOG", "0").strip() not in ("", "0", "false"):
                    from jinx.logger.file_logger import append_line
                    from jinx.log_paths import BLUE_WHISPERS
                    await append_line(
                        BLUE_WHISPERS,
                        f"[rate_limiter] adapted limit to {new_limit:.1f} req/s "
                        f"(util={utilization:.2%}, congested={congested})"
                    )
            except Exception:
                pass
            
            # Periodically save
            if self.total_requests % 50 == 0:
                await self._save_state()
    
    async def acquire(self, latency_hint: Optional[float] = None) -> bool:
        """Try to acquire permission for a request.
        
        Args:
            latency_hint: Optional latency of previous request for congestion detection
        
        Returns:
            True if allowed, False if rate limited
        """
        async with self._lock:
            now = time.time()
            self.total_requests += 1
            
            # Record latency if provided
            if latency_hint is not None and latency_hint > 0:
                self.latencies.append(latency_hint)
            
            # Clean old requests
            self._clean_old_requests(now)
            
            # Count requests in current window
            window_requests = len(self.requests)
            
            # Check if we can allow this request
            if window_requests >= self.current_limit * self.window_seconds:
                self.total_denied += 1
                
                # Adapt on denial
                await self._adapt_limit(now)
                
                return False
            
            # Allow request
            self.requests.append(now)
            self.total_allowed += 1
            
            # Periodic adaptation
            await self._adapt_limit(now)
            
            return True
    
    async def wait_for_slot(self, timeout: Optional[float] = None) -> bool:
        """Wait until a slot is available.
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            True if acquired, False if timed out
        """
        start = time.time()
        
        while True:
            if await self.acquire():
                return True
            
            # Check timeout
            if timeout and (time.time() - start) >= timeout:
                return False
            
            # Calculate wait time
            async with self._lock:
                now = time.time()
                self._clean_old_requests(now)
                
                if not self.requests:
                    wait = 0.0
                else:
                    oldest = self.requests[0]
                    wait = max(0.0, oldest + self.window_seconds - now)
            
            # Wait with small buffer
            await asyncio.sleep(min(0.1, wait + 0.01))
    
    def get_state(self) -> RateLimitState:
        """Get current rate limiter state."""
        now = time.time()
        
        window_requests = sum(1 for t in self.requests if t >= now - self.window_seconds)
        avg_latency = sum(self.latencies) / max(1, len(self.latencies)) if self.latencies else 0.0
        
        return RateLimitState(
            current_limit=self.current_limit,
            window_start=now - self.window_seconds,
            request_count=window_requests,
            denied_count=self.total_denied,
            avg_latency=avg_latency
        )
    
    def get_stats(self) -> Dict[str, object]:
        """Get statistics."""
        total = self.total_requests
        allowed_rate = self.total_allowed / total if total > 0 else 0.0
        
        return {
            'current_limit': self.current_limit,
            'total_requests': total,
            'total_allowed': self.total_allowed,
            'total_denied': self.total_denied,
            'allowed_rate': allowed_rate,
            'avg_latency': sum(self.latencies) / max(1, len(self.latencies)) if self.latencies else 0.0
        }


# Global limiters by context
_limiters: Dict[str, AdaptiveRateLimiter] = {}
_limiters_lock = asyncio.Lock()


async def get_rate_limiter(context: str = "default") -> AdaptiveRateLimiter:
    """Get or create rate limiter for context."""
    async with _limiters_lock:
        if context not in _limiters:
            _limiters[context] = AdaptiveRateLimiter(state_path=f"log/rate_limiter_{context}.json")
        return _limiters[context]


async def acquire_rate_limit(context: str = "default", latency_hint: Optional[float] = None) -> bool:
    """Acquire rate limit slot."""
    limiter = await get_rate_limiter(context)
    return await limiter.acquire(latency_hint)


__all__ = [
    "AdaptiveRateLimiter",
    "RateLimitState",
    "get_rate_limiter",
    "acquire_rate_limit",
]
