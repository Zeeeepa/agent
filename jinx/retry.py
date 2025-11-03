"""Advanced retry helper utilities with circuit breaker and adaptive backoff."""

from __future__ import annotations

import asyncio
import random
import time
from enum import Enum
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar, Generic
from .logging_service import bomb_log

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3


class CircuitBreaker(Generic[T]):
    """Advanced circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by failing fast when a service is down.
    Automatically attempts recovery after a timeout period.
    """
    
    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function through circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout elapsed
                if (time.time() - self.last_failure_time) >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise RuntimeError("Circuit breaker is OPEN, rejecting call")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise RuntimeError("Circuit breaker HALF_OPEN limit reached")
                self.half_open_calls += 1
        
        try:
            result = await func()
            # Success - update state
            async with self._lock:
                self.success_count += 1
                if self.state == CircuitState.HALF_OPEN:
                    # Successful test, transition to closed
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            return result
        except Exception as e:
            # Failure - update state
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.state == CircuitState.HALF_OPEN:
                    # Failed during test, reopen circuit
                    self.state = CircuitState.OPEN
                elif self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
            raise


async def detonate_payload(
    pyro: Callable[[], Awaitable[T]],
    retries: int = 2,
    delay: float = 3,
    *,
    timeout: float | None = None,
    jitter: float = 0.0,
    exponential_backoff: bool = True,
    max_delay: float = 60.0,
) -> T:
    """Execute an async callable with advanced retry logic.

    Features:
    - Exponential backoff with jitter
    - Optional timeout per attempt
    - Configurable max delay cap
    - Detailed failure tracking

    Parameters
    ----------
    pyro : Callable[[], Awaitable[T]]
        Async function to invoke.
    retries : int
        Number of attempts before giving up.
    delay : float
        Base delay in seconds between attempts.
    timeout : float | None
        Optional timeout per attempt.
    jitter : float
        Random jitter range to prevent thundering herd.
    exponential_backoff : bool
        Use exponential backoff strategy (recommended).
    max_delay : float
        Maximum delay between retries (caps exponential growth).
    """
    # Guarantee at least one attempt
    attempts = max(1, int(retries))
    last_exception: Exception | None = None
    
    for attempt in range(attempts):
        try:
            if timeout is not None:
                result = await asyncio.wait_for(pyro(), timeout=timeout)
            else:
                result = await pyro()
            
            # Success - log recovery if we had failures
            if attempt > 0:
                await bomb_log(f"Recovery successful after {attempt} failures")
            return result
            
        except Exception as e:
            last_exception = e
            await bomb_log(f"Spiking the loop: Detonating again: {e} (attempt {attempt + 1}/{attempts})")
            
            if attempt < attempts - 1:
                # Calculate backoff delay
                if exponential_backoff:
                    # Exponential backoff: delay * 2^attempt
                    backoff_delay = min(delay * (2 ** attempt), max_delay)
                else:
                    backoff_delay = delay
                
                # Apply jitter to reduce thundering herd
                if jitter > 0:
                    jitter_amount = random.uniform(-jitter, jitter)
                    sleep_for = max(0.1, backoff_delay + jitter_amount)
                else:
                    sleep_for = backoff_delay
                
                await bomb_log(f"Backing off for {sleep_for:.2f}s before retry")
                await asyncio.sleep(sleep_for)
            else:
                await bomb_log("System fracturing: Max retries burned.")
    
    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RuntimeError("detonate_payload: no attempts executed")


async def detonate_with_circuit_breaker(
    pyro: Callable[[], Awaitable[T]],
    breaker: CircuitBreaker[T] | None = None,
    retries: int = 2,
    delay: float = 3,
    **kwargs,
) -> T:
    """Execute function with both circuit breaker and retry logic.
    
    Combines fault tolerance patterns for maximum resilience.
    """
    if breaker is None:
        breaker = CircuitBreaker[T]()
    
    async def _wrapped_call() -> T:
        return await breaker.call(pyro)
    
    return await detonate_payload(_wrapped_call, retries=retries, delay=delay, **kwargs)
