"""System Health Check - Production health monitoring and diagnostics.

Features:
- Component health checks
- Dependency verification
- Performance benchmarks
- Resource availability checks
- Automatic recovery triggers
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a component."""
    component: str
    status: HealthStatus
    latency_ms: float
    message: str
    details: Dict[str, Any]
    timestamp: float


class SystemHealth:
    """
    Production health check system.
    
    Monitors all critical components and reports overall system health.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Component health cache
        self._health_cache: Dict[str, ComponentHealth] = {}
        
        # Cache TTL
        self._cache_ttl = 30.0  # 30 seconds
        
        # Critical components
        self._critical_components = [
            'ml_orchestrator',
            'monitoring',
            'cache',
            'embeddings'
        ]
    
    async def check_all(self) -> Dict[str, Any]:
        """
        Check health of all components.
        
        Returns:
            {
                'overall_status': str,
                'components': dict,
                'timestamp': float,
                'critical_failures': list
            }
        """
        
        async with self._lock:
            components = {}
            critical_failures = []
            
            # Check ML orchestrator
            ml_health = await self._check_ml_orchestrator()
            components['ml_orchestrator'] = ml_health
            
            if ml_health.status == HealthStatus.UNHEALTHY:
                critical_failures.append('ml_orchestrator')
            
            # Check monitoring
            monitor_health = await self._check_monitoring()
            components['monitoring'] = monitor_health
            
            # Check cache
            cache_health = await self._check_cache()
            components['cache'] = cache_health
            
            if cache_health.status == HealthStatus.UNHEALTHY:
                critical_failures.append('cache')
            
            # Check embeddings
            embed_health = await self._check_embeddings()
            components['embeddings'] = embed_health
            
            if embed_health.status == HealthStatus.UNHEALTHY:
                critical_failures.append('embeddings')
            
            # Check performance optimizer
            perf_health = await self._check_performance_optimizer()
            components['performance_optimizer'] = perf_health
            
            # Check auto scaler
            scaler_health = await self._check_auto_scaler()
            components['auto_scaler'] = scaler_health
            
            # Determine overall status
            overall_status = self._compute_overall_status(components, critical_failures)
            
            return {
                'overall_status': overall_status.value,
                'components': {
                    name: {
                        'status': health.status.value,
                        'latency_ms': health.latency_ms,
                        'message': health.message,
                        'details': health.details
                    }
                    for name, health in components.items()
                },
                'timestamp': time.time(),
                'critical_failures': critical_failures
            }
    
    async def _check_ml_orchestrator(self) -> ComponentHealth:
        """Check ML orchestrator health."""
        
        start = time.time()
        
        try:
            from .ml_orchestrator import get_ml_orchestrator
            
            orchestrator = await get_ml_orchestrator()
            health = await orchestrator.health_check()
            
            latency = (time.time() - start) * 1000
            
            if all(health.values()):
                return ComponentHealth(
                    component='ml_orchestrator',
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message='All components available',
                    details=health,
                    timestamp=time.time()
                )
            else:
                return ComponentHealth(
                    component='ml_orchestrator',
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message='Some components unavailable',
                    details=health,
                    timestamp=time.time()
                )
        
        except Exception as e:
            return ComponentHealth(
                component='ml_orchestrator',
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=f'Error: {str(e)}',
                details={},
                timestamp=time.time()
            )
    
    async def _check_monitoring(self) -> ComponentHealth:
        """Check monitoring system health."""
        
        start = time.time()
        
        try:
            from .ml_monitoring import get_ml_monitoring
            
            monitor = await get_ml_monitoring()
            summary = monitor.get_metrics_summary()
            
            latency = (time.time() - start) * 1000
            
            return ComponentHealth(
                component='monitoring',
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message='Monitoring active',
                details={'predictions': summary.get('total_predictions', 0)},
                timestamp=time.time()
            )
        
        except Exception as e:
            return ComponentHealth(
                component='monitoring',
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=f'Error: {str(e)}',
                details={},
                timestamp=time.time()
            )
    
    async def _check_cache(self) -> ComponentHealth:
        """Check cache health."""
        
        start = time.time()
        
        try:
            from jinx.micro.embeddings.semantic_cache import get_embedding_cache
            
            cache = await get_embedding_cache()
            stats = cache.get_stats()
            
            latency = (time.time() - start) * 1000
            
            # Check hit rate
            hit_rate = stats.get('hit_rate', 0)
            
            if hit_rate > 0.5:
                status = HealthStatus.HEALTHY
                message = f'Cache hit rate: {hit_rate:.1%}'
            elif hit_rate > 0.2:
                status = HealthStatus.DEGRADED
                message = f'Low hit rate: {hit_rate:.1%}'
            else:
                status = HealthStatus.DEGRADED
                message = f'Very low hit rate: {hit_rate:.1%}'
            
            return ComponentHealth(
                component='cache',
                status=status,
                latency_ms=latency,
                message=message,
                details=stats,
                timestamp=time.time()
            )
        
        except Exception as e:
            return ComponentHealth(
                component='cache',
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=f'Error: {str(e)}',
                details={},
                timestamp=time.time()
            )
    
    async def _check_embeddings(self) -> ComponentHealth:
        """Check embeddings service health."""
        
        start = time.time()
        
        try:
            from jinx.micro.embeddings.pipeline import embed_text
            
            # Quick test
            result = await embed_text("health check", source='health_check', persist=False)
            
            latency = (time.time() - start) * 1000
            
            if result and 'embedding' in result:
                return ComponentHealth(
                    component='embeddings',
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message='Embeddings service responsive',
                    details={'test_passed': True},
                    timestamp=time.time()
                )
            else:
                return ComponentHealth(
                    component='embeddings',
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message='Embeddings service degraded',
                    details={'test_passed': False},
                    timestamp=time.time()
                )
        
        except Exception as e:
            return ComponentHealth(
                component='embeddings',
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=f'Error: {str(e)}',
                details={},
                timestamp=time.time()
            )
    
    async def _check_performance_optimizer(self) -> ComponentHealth:
        """Check performance optimizer health."""
        
        start = time.time()
        
        try:
            from .performance_optimizer import get_performance_optimizer
            
            optimizer = await get_performance_optimizer()
            stats = optimizer.get_stats()
            
            latency = (time.time() - start) * 1000
            
            return ComponentHealth(
                component='performance_optimizer',
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message='Performance optimizer active',
                details=stats,
                timestamp=time.time()
            )
        
        except Exception as e:
            return ComponentHealth(
                component='performance_optimizer',
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=f'Error: {str(e)}',
                details={},
                timestamp=time.time()
            )
    
    async def _check_auto_scaler(self) -> ComponentHealth:
        """Check auto scaler health."""
        
        start = time.time()
        
        try:
            from .auto_scaler import get_auto_scaler
            
            scaler = await get_auto_scaler()
            stats = scaler.get_stats()
            
            latency = (time.time() - start) * 1000
            
            # Check if circuit breaker is open
            if stats.get('circuit_open', False):
                return ComponentHealth(
                    component='auto_scaler',
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message='Circuit breaker OPEN',
                    details=stats,
                    timestamp=time.time()
                )
            
            return ComponentHealth(
                component='auto_scaler',
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message=f"State: {stats.get('state', 'unknown')}",
                details=stats,
                timestamp=time.time()
            )
        
        except Exception as e:
            return ComponentHealth(
                component='auto_scaler',
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=f'Error: {str(e)}',
                details={},
                timestamp=time.time()
            )
    
    def _compute_overall_status(
        self,
        components: Dict[str, ComponentHealth],
        critical_failures: List[str]
    ) -> HealthStatus:
        """Compute overall system status."""
        
        if critical_failures:
            return HealthStatus.UNHEALTHY
        
        # Check if any component is unhealthy
        for health in components.values():
            if health.status == HealthStatus.UNHEALTHY:
                if health.component in self._critical_components:
                    return HealthStatus.UNHEALTHY
        
        # Check if multiple components degraded
        degraded_count = sum(
            1 for h in components.values()
            if h.status == HealthStatus.DEGRADED
        )
        
        if degraded_count >= 2:
            return HealthStatus.DEGRADED
        elif degraded_count == 1:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY


# Singleton
_system_health: Optional[SystemHealth] = None
_health_lock = asyncio.Lock()


async def get_system_health() -> SystemHealth:
    """Get singleton system health."""
    global _system_health
    if _system_health is None:
        async with _health_lock:
            if _system_health is None:
                _system_health = SystemHealth()
    return _system_health


async def health_check() -> Dict[str, Any]:
    """Quick health check API."""
    health = await get_system_health()
    return await health.check_all()


__all__ = [
    "SystemHealth",
    "HealthStatus",
    "ComponentHealth",
    "get_system_health",
    "health_check",
]
