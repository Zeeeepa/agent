"""Real-time Performance Monitor для всех brain систем.

Отслеживает метрики, детектирует аномалии и автоматически реагирует.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    system_name: str
    timestamp: float
    latency_ms: float
    success_rate: float
    throughput: float  # operations per second
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class Anomaly:
    """Detected performance anomaly."""
    system_name: str
    metric_name: str
    current_value: float
    expected_value: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: float
    auto_corrected: bool


class PerformanceMonitor:
    """Real-time мониторинг с автоматическим обнаружением и исправлением проблем."""
    
    def __init__(self, state_path: str = "log/performance_monitor.json"):
        self.state_path = state_path
        
        # Real-time metrics per system
        self.metrics: Dict[str, deque[PerformanceMetrics]] = defaultdict(lambda: deque(maxlen=100))
        
        # Detected anomalies
        self.anomalies: deque[Anomaly] = deque(maxlen=500)
        
        # Baseline metrics (learned)
        self.baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Alert thresholds (adaptive)
        self.thresholds: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)  # (min, max)
        
        # Auto-correction actions
        self.corrections: Dict[str, int] = defaultdict(int)
        
        # Performance trends
        self.trends: Dict[str, str] = {}  # 'improving', 'degrading', 'stable'
        
        self._lock = asyncio.Lock()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        self._load_state()
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self) -> None:
        """Setup default alert thresholds."""
        default_systems = [
            'adaptive_retrieval', 'threshold_learner', 'query_classifier',
            'context_optimizer', 'semantic_router', 'intelligent_planner',
            'prompt_optimizer', 'error_predictor', 'self_healing',
            'rate_limiter', 'predictive_cache', 'outcome_tracker',
            'orchestrator', 'auto_tuner', 'learning_coordinator', 'performance_monitor'
        ]
        
        for system in default_systems:
            # Latency thresholds
            self.thresholds[system]['latency_ms'] = (0.0, 1000.0)
            
            # Success rate thresholds
            self.thresholds[system]['success_rate'] = (0.5, 1.0)
            
            # Throughput thresholds
            self.thresholds[system]['throughput'] = (0.1, 1000.0)
    
    def _load_state(self) -> None:
        """Load monitor state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore baselines
                for system, baseline in data.get('baselines', {}).items():
                    self.baselines[system] = baseline
                
                # Restore thresholds
                for system, thresh in data.get('thresholds', {}).items():
                    self.thresholds[system] = {
                        k: tuple(v) for k, v in thresh.items()
                    }
                
                # Restore trends
                self.trends = data.get('trends', {})
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist monitor state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize thresholds
                thresholds_data = {}
                for system, thresh in self.thresholds.items():
                    thresholds_data[system] = {k: list(v) for k, v in thresh.items()}
                
                data = {
                    'baselines': dict(self.baselines),
                    'thresholds': thresholds_data,
                    'trends': self.trends,
                    'corrections': dict(self.corrections),
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def start(self) -> None:
        """Start performance monitoring."""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop(self) -> None:
        """Stop performance monitoring."""
        async with self._lock:
            self._running = False
            
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
    
    async def record_metrics(
        self,
        system_name: str,
        latency_ms: float,
        success: bool,
        operation_count: int = 1
    ) -> None:
        """Record real-time metrics."""
        async with self._lock:
            now = time.time()
            
            # Calculate throughput
            recent_metrics = list(self.metrics[system_name])[-10:]
            if recent_metrics:
                time_span = now - recent_metrics[0].timestamp
                throughput = len(recent_metrics) / max(0.1, time_span)
            else:
                throughput = 1.0
            
            # Calculate success rate
            if recent_metrics:
                successes = sum(1 for m in recent_metrics if m.success_rate > 0.5)
                success_rate = successes / len(recent_metrics)
            else:
                success_rate = 1.0 if success else 0.0
            
            # Memory and CPU (placeholder - would need psutil)
            memory_mb = 50.0  # Placeholder
            cpu_percent = 10.0  # Placeholder
            
            metrics = PerformanceMetrics(
                system_name=system_name,
                timestamp=now,
                latency_ms=latency_ms,
                success_rate=success_rate,
                throughput=throughput,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent
            )
            
            self.metrics[system_name].append(metrics)
            
            # Check for anomalies
            await self._check_anomalies(system_name, metrics)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)
                continue
            
            try:
                # Update baselines
                await self._update_baselines()
                
                # Detect trends
                await self._detect_trends()
                
                # Auto-correct severe anomalies
                await self._auto_correct_anomalies()
                
                # Save state periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    await self._save_state()
                
            except Exception:
                pass
    
    async def _update_baselines(self) -> None:
        """Update baseline metrics from recent history."""
        try:
            for system_name, metrics_list in self.metrics.items():
                if len(metrics_list) < 10:
                    continue
                
                recent = list(metrics_list)[-20:]
                
                # Compute averages
                avg_latency = sum(m.latency_ms for m in recent) / len(recent)
                avg_success = sum(m.success_rate for m in recent) / len(recent)
                avg_throughput = sum(m.throughput for m in recent) / len(recent)
                
                # Update baselines with EMA
                alpha = 0.1
                
                current_baselines = self.baselines[system_name]
                current_baselines['latency_ms'] = (
                    alpha * avg_latency + (1 - alpha) * current_baselines.get('latency_ms', avg_latency)
                )
                current_baselines['success_rate'] = (
                    alpha * avg_success + (1 - alpha) * current_baselines.get('success_rate', avg_success)
                )
                current_baselines['throughput'] = (
                    alpha * avg_throughput + (1 - alpha) * current_baselines.get('throughput', avg_throughput)
                )
        except Exception:
            pass
    
    async def _check_anomalies(self, system_name: str, metrics: PerformanceMetrics) -> None:
        """Check for performance anomalies."""
        try:
            baseline = self.baselines.get(system_name, {})
            thresholds = self.thresholds.get(system_name, {})
            
            # Check latency
            if 'latency_ms' in baseline:
                expected_latency = baseline['latency_ms']
                min_lat, max_lat = thresholds.get('latency_ms', (0.0, 1000.0))
                
                if metrics.latency_ms > max_lat or metrics.latency_ms > expected_latency * 3:
                    severity = 'critical' if metrics.latency_ms > expected_latency * 5 else 'high'
                    
                    anomaly = Anomaly(
                        system_name=system_name,
                        metric_name='latency_ms',
                        current_value=metrics.latency_ms,
                        expected_value=expected_latency,
                        severity=severity,
                        timestamp=time.time(),
                        auto_corrected=False
                    )
                    
                    self.anomalies.append(anomaly)
            
            # Check success rate
            if 'success_rate' in baseline:
                expected_success = baseline['success_rate']
                
                if metrics.success_rate < expected_success * 0.7:
                    severity = 'critical' if metrics.success_rate < 0.3 else 'high'
                    
                    anomaly = Anomaly(
                        system_name=system_name,
                        metric_name='success_rate',
                        current_value=metrics.success_rate,
                        expected_value=expected_success,
                        severity=severity,
                        timestamp=time.time(),
                        auto_corrected=False
                    )
                    
                    self.anomalies.append(anomaly)
        except Exception:
            pass
    
    async def _detect_trends(self) -> None:
        """Detect performance trends."""
        try:
            for system_name, metrics_list in self.metrics.items():
                if len(metrics_list) < 20:
                    continue
                
                recent = list(metrics_list)[-10:]
                older = list(metrics_list)[-20:-10]
                
                # Compare success rates
                recent_success = sum(m.success_rate for m in recent) / len(recent)
                older_success = sum(m.success_rate for m in older) / len(older)
                
                # Determine trend
                if recent_success > older_success + 0.1:
                    self.trends[system_name] = 'improving'
                elif recent_success < older_success - 0.1:
                    self.trends[system_name] = 'degrading'
                else:
                    self.trends[system_name] = 'stable'
        except Exception:
            pass
    
    async def _auto_correct_anomalies(self) -> None:
        """Automatically correct severe anomalies."""
        try:
            # Get recent critical anomalies
            recent_critical = [
                a for a in list(self.anomalies)[-20:]
                if a.severity == 'critical' and not a.auto_corrected
                and (time.time() - a.timestamp) < 60.0
            ]
            
            for anomaly in recent_critical:
                await self._apply_correction(anomaly)
                anomaly.auto_corrected = True
        except Exception:
            pass
    
    async def _apply_correction(self, anomaly: Anomaly) -> None:
        """Apply automatic correction for anomaly."""
        try:
            system = anomaly.system_name
            metric = anomaly.metric_name
            
            # System-specific corrections
            if system == 'adaptive_retrieval' and metric == 'latency_ms':
                # Reduce k to speed up
                from jinx.micro.brain import emit_learning_event
                await emit_learning_event('performance_monitor', 'optimization', {
                    'system': system,
                    'action': 'reduce_k',
                    'reason': 'high_latency'
                })
            
            elif system == 'rate_limiter' and metric == 'success_rate':
                # Increase rate limit
                from jinx.micro.brain import emit_learning_event
                await emit_learning_event('performance_monitor', 'optimization', {
                    'system': system,
                    'action': 'increase_limit',
                    'reason': 'low_success_rate'
                })
            
            self.corrections[f"{system}_{metric}"] += 1
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        systems_monitored = len(self.metrics)
        total_metrics = sum(len(m) for m in self.metrics.values())
        
        recent_anomalies = [
            a for a in list(self.anomalies)[-20:]
            if (time.time() - a.timestamp) < 300.0  # Last 5 minutes
        ]
        
        return {
            'systems_monitored': systems_monitored,
            'total_metrics_recorded': total_metrics,
            'recent_anomalies': len(recent_anomalies),
            'critical_anomalies': len([a for a in recent_anomalies if a.severity == 'critical']),
            'auto_corrections': sum(self.corrections.values()),
            'trends': self.trends,
            'running': self._running
        }
    
    async def get_system_health(self, system_name: str) -> Dict[str, Any]:
        """Get detailed health report for a system."""
        metrics_list = list(self.metrics.get(system_name, []))
        
        if not metrics_list:
            return {'status': 'unknown', 'message': 'No metrics available'}
        
        recent = metrics_list[-10:]
        
        avg_latency = sum(m.latency_ms for m in recent) / len(recent)
        avg_success = sum(m.success_rate for m in recent) / len(recent)
        
        # Determine health status
        if avg_success > 0.9 and avg_latency < 500:
            status = 'healthy'
        elif avg_success > 0.7 and avg_latency < 1000:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'avg_latency_ms': avg_latency,
            'avg_success_rate': avg_success,
            'trend': self.trends.get(system_name, 'unknown'),
            'metrics_count': len(metrics_list)
        }


# Singleton
_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = asyncio.Lock()


async def get_performance_monitor() -> PerformanceMonitor:
    """Get singleton performance monitor."""
    global _monitor
    if _monitor is None:
        async with _monitor_lock:
            if _monitor is None:
                _monitor = PerformanceMonitor()
                # Auto-start
                await _monitor.start()
    return _monitor


async def record_system_metrics(system: str, latency_ms: float, success: bool) -> None:
    """Record metrics for a system."""
    monitor = await get_performance_monitor()
    await monitor.record_metrics(system, latency_ms, success)


__all__ = [
    "PerformanceMonitor",
    "get_performance_monitor",
    "record_system_metrics",
]
