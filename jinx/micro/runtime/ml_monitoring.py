"""ML Monitoring - Real-time quality tracking and alerting.

Features:
- Performance metrics tracking
- Quality degradation detection
- Automatic alerting
- Drift detection
- Anomaly detection
- Dashboard-ready metrics
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import os


@dataclass
class PredictionMetric:
    """Single prediction metric."""
    timestamp: float
    task_type: str
    confidence: float
    latency_ms: float
    success: bool
    quality_score: float


@dataclass
class Alert:
    """System alert."""
    timestamp: float
    severity: str  # 'warning', 'error', 'critical'
    category: str
    message: str
    metrics: Dict[str, Any]


class MLMonitoring:
    """
    Real-time ML system monitoring.
    
    Tracks:
    - Prediction quality
    - Latency
    - Confidence distribution
    - Quality degradation
    - Anomalies
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Metrics buffer (last N predictions)
        self._metrics: deque[PredictionMetric] = deque(maxlen=10000)
        
        # Per-task metrics
        self._task_metrics: Dict[str, deque[PredictionMetric]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Alerts
        self._alerts: deque[Alert] = deque(maxlen=1000)
        
        # Baseline metrics (for drift detection)
        self._baseline: Optional[Dict[str, float]] = None
        
        # Alert thresholds
        self._thresholds = {
            'min_confidence': 0.5,
            'max_latency_ms': 1000,
            'min_quality': 0.6,
            'min_success_rate': 0.8,
            'drift_threshold': 0.15
        }
        
        # Monitoring state
        self._monitoring_active = True
        self._last_alert_time = {}  # Debounce alerts
        
        # Storage
        self._storage_path = '.jinx/monitoring/ml_metrics.json'
    
    async def record_prediction(
        self,
        task_type: str,
        confidence: float,
        latency_ms: float,
        success: bool,
        quality_score: float
    ):
        """Record prediction metrics."""
        
        async with self._lock:
            metric = PredictionMetric(
                timestamp=time.time(),
                task_type=task_type,
                confidence=confidence,
                latency_ms=latency_ms,
                success=success,
                quality_score=quality_score
            )
            
            # Add to global buffer
            self._metrics.append(metric)
            
            # Add to task-specific buffer
            self._task_metrics[task_type].append(metric)
            
            # Check for issues
            await self._check_for_issues(metric)
    
    async def _check_for_issues(self, metric: PredictionMetric):
        """Check for quality issues and create alerts."""
        
        issues = []
        
        # Low confidence
        if metric.confidence < self._thresholds['min_confidence']:
            issues.append(('warning', 'low_confidence', 
                          f"Low confidence: {metric.confidence:.2f}"))
        
        # High latency
        if metric.latency_ms > self._thresholds['max_latency_ms']:
            issues.append(('warning', 'high_latency',
                          f"High latency: {metric.latency_ms:.0f}ms"))
        
        # Low quality
        if metric.quality_score < self._thresholds['min_quality']:
            issues.append(('warning', 'low_quality',
                          f"Low quality: {metric.quality_score:.2f}"))
        
        # Failure
        if not metric.success:
            issues.append(('error', 'prediction_failure',
                          f"Prediction failed for {metric.task_type}"))
        
        # Create alerts (with debouncing)
        for severity, category, message in issues:
            await self._create_alert(severity, category, message, metric)
    
    async def _create_alert(
        self,
        severity: str,
        category: str,
        message: str,
        metric: PredictionMetric
    ):
        """Create alert with debouncing."""
        
        # Debounce: don't alert same category within 60s
        last_alert = self._last_alert_time.get(category, 0)
        if time.time() - last_alert < 60:
            return
        
        alert = Alert(
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=message,
            metrics={
                'task_type': metric.task_type,
                'confidence': metric.confidence,
                'latency_ms': metric.latency_ms,
                'quality_score': metric.quality_score
            }
        )
        
        self._alerts.append(alert)
        self._last_alert_time[category] = time.time()
        
        # Log critical alerts
        if severity == 'critical':
            try:
                from jinx.micro.logger.debug_logger import debug_log
                await debug_log(f"CRITICAL ALERT: {message}", "ML_MONITOR")
            except Exception:
                pass
    
    async def check_drift(self) -> bool:
        """
        Check for distribution drift.
        
        Returns True if significant drift detected.
        """
        
        async with self._lock:
            if len(self._metrics) < 100:
                return False
            
            # Get recent metrics
            recent = list(self._metrics)[-100:]
            
            # Calculate current stats
            current_stats = {
                'avg_confidence': np.mean([m.confidence for m in recent]),
                'avg_latency': np.mean([m.latency_ms for m in recent]),
                'avg_quality': np.mean([m.quality_score for m in recent]),
                'success_rate': np.mean([float(m.success) for m in recent])
            }
            
            # First time - establish baseline
            if self._baseline is None:
                self._baseline = current_stats
                return False
            
            # Check for drift
            drift_detected = False
            
            for key in ['avg_confidence', 'avg_quality', 'success_rate']:
                baseline_val = self._baseline.get(key, 0)
                current_val = current_stats.get(key, 0)
                
                if baseline_val > 0:
                    drift = abs(current_val - baseline_val) / baseline_val
                    
                    if drift > self._thresholds['drift_threshold']:
                        await self._create_alert(
                            'warning',
                            'drift_detected',
                            f"Drift detected in {key}: {drift:.2%}",
                            recent[-1]
                        )
                        drift_detected = True
            
            return drift_detected
    
    async def detect_anomalies(self) -> List[PredictionMetric]:
        """
        Detect anomalous predictions using statistical methods.
        
        Returns list of anomalous predictions.
        """
        
        async with self._lock:
            if len(self._metrics) < 50:
                return []
            
            recent = list(self._metrics)[-1000:]
            
            # Extract features
            confidences = np.array([m.confidence for m in recent])
            latencies = np.array([m.latency_ms for m in recent])
            qualities = np.array([m.quality_score for m in recent])
            
            # Z-score based anomaly detection
            anomalies = []
            
            for i, metric in enumerate(recent):
                # Confidence anomaly
                conf_z = abs((metric.confidence - np.mean(confidences)) / 
                            (np.std(confidences) + 1e-10))
                
                # Latency anomaly
                lat_z = abs((metric.latency_ms - np.mean(latencies)) /
                           (np.std(latencies) + 1e-10))
                
                # Quality anomaly
                qual_z = abs((metric.quality_score - np.mean(qualities)) /
                            (np.std(qualities) + 1e-10))
                
                # If any feature is >3 std away, it's anomalous
                if max(conf_z, lat_z, qual_z) > 3.0:
                    anomalies.append(metric)
            
            return anomalies
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        
        if not self._metrics:
            return {}
        
        recent = list(self._metrics)[-1000:]
        
        return {
            'total_predictions': len(self._metrics),
            'recent_predictions': len(recent),
            'avg_confidence': float(np.mean([m.confidence for m in recent])),
            'avg_latency_ms': float(np.mean([m.latency_ms for m in recent])),
            'avg_quality': float(np.mean([m.quality_score for m in recent])),
            'success_rate': float(np.mean([float(m.success) for m in recent])),
            'p95_latency': float(np.percentile([m.latency_ms for m in recent], 95)),
            'p99_latency': float(np.percentile([m.latency_ms for m in recent], 99)),
            'task_distribution': self._get_task_distribution(recent)
        }
    
    def _get_task_distribution(self, metrics: List[PredictionMetric]) -> Dict[str, int]:
        """Get distribution of task types."""
        
        dist = defaultdict(int)
        for m in metrics:
            dist[m.task_type] += 1
        
        return dict(dist)
    
    def get_task_metrics(self, task_type: str) -> Dict[str, Any]:
        """Get metrics for specific task type."""
        
        task_metrics = list(self._task_metrics.get(task_type, []))
        
        if not task_metrics:
            return {}
        
        return {
            'total_predictions': len(task_metrics),
            'avg_confidence': float(np.mean([m.confidence for m in task_metrics])),
            'avg_latency_ms': float(np.mean([m.latency_ms for m in task_metrics])),
            'avg_quality': float(np.mean([m.quality_score for m in task_metrics])),
            'success_rate': float(np.mean([float(m.success) for m in task_metrics]))
        }
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        
        return [
            {
                'timestamp': alert.timestamp,
                'severity': alert.severity,
                'category': alert.category,
                'message': alert.message,
                'metrics': alert.metrics
            }
            for alert in list(self._alerts)[-limit:]
        ]
    
    async def save_to_disk(self):
        """Save metrics to disk."""
        
        async with self._lock:
            try:
                os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
                
                data = {
                    'summary': self.get_metrics_summary(),
                    'baseline': self._baseline,
                    'alerts': self.get_recent_alerts(100),
                    'timestamp': time.time()
                }
                
                with open(self._storage_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            except Exception:
                pass
    
    async def start_monitoring_loop(self):
        """Start background monitoring loop."""
        
        while self._monitoring_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for drift
                await self.check_drift()
                
                # Detect anomalies
                anomalies = await self.detect_anomalies()
                
                if len(anomalies) > 5:
                    await self._create_alert(
                        'warning',
                        'anomalies_detected',
                        f"Detected {len(anomalies)} anomalous predictions",
                        anomalies[0] if anomalies else None
                    )
                
                # Save metrics
                await self.save_to_disk()
            
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    def stop(self):
        """Stop monitoring."""
        self._monitoring_active = False


# Singleton
_ml_monitoring: Optional[MLMonitoring] = None
_monitor_lock = asyncio.Lock()


async def get_ml_monitoring() -> MLMonitoring:
    """Get singleton ML monitoring."""
    global _ml_monitoring
    if _ml_monitoring is None:
        async with _monitor_lock:
            if _ml_monitoring is None:
                _ml_monitoring = MLMonitoring()
                # Start monitoring loop
                asyncio.create_task(_ml_monitoring.start_monitoring_loop())
    return _ml_monitoring


__all__ = [
    "MLMonitoring",
    "PredictionMetric",
    "Alert",
    "get_ml_monitoring",
]
