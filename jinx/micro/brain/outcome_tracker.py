"""Global outcome tracking system for continuous learning.

Tracks outcomes of all major operations to feed ML components:
- Code execution success/failure
- Retrieval hit rates
- Context usefulness
- Response quality
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Outcome:
    """Generic outcome record."""
    operation: str  # 'code_exec', 'retrieval', 'context_build', 'response'
    success: bool
    metadata: Dict[str, Any]
    timestamp: float
    latency_ms: float = 0.0


class OutcomeTracker:
    """Central outcome tracking for system-wide learning."""
    
    def __init__(self, state_path: str = "log/outcomes.jsonl"):
        self.state_path = state_path
        self.recent: deque[Outcome] = deque(maxlen=500)
        self._lock = asyncio.Lock()
        
        # Aggregated metrics
        self.metrics: Dict[str, Dict[str, Any]] = {
            'code_exec': {'total': 0, 'success': 0, 'failures': 0},
            'retrieval': {'total': 0, 'useful': 0, 'empty': 0},
            'context_build': {'total': 0, 'effective': 0, 'ineffective': 0},
            'response': {'total': 0, 'quality_sum': 0.0},
        }
    
    async def record(
        self,
        operation: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0
    ) -> None:
        """Record outcome."""
        async with self._lock:
            outcome = Outcome(
                operation=operation,
                success=success,
                metadata=metadata or {},
                timestamp=time.time(),
                latency_ms=latency_ms
            )
            
            self.recent.append(outcome)
            
            # Update metrics
            if operation in self.metrics:
                self.metrics[operation]['total'] += 1
                
                if operation == 'code_exec':
                    if success:
                        self.metrics[operation]['success'] += 1
                    else:
                        self.metrics[operation]['failures'] += 1
                
                elif operation == 'retrieval':
                    hit_count = metadata.get('hits', 0) if metadata else 0
                    if hit_count > 0:
                        self.metrics[operation]['useful'] += 1
                    else:
                        self.metrics[operation]['empty'] += 1
                
                elif operation == 'context_build':
                    if success:
                        self.metrics[operation]['effective'] += 1
                    else:
                        self.metrics[operation]['ineffective'] += 1
                
                elif operation == 'response':
                    quality = metadata.get('quality', 0.5) if metadata else 0.5
                    self.metrics[operation]['quality_sum'] += quality
            
            # Periodically flush to disk
            if len(self.recent) % 20 == 0:
                asyncio.create_task(self._flush())
            
            # Trigger learning updates
            asyncio.create_task(self._propagate_learning(outcome))
    
    async def _flush(self) -> None:
        """Flush recent outcomes to disk."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                with open(self.state_path, 'a', encoding='utf-8') as f:
                    for out in list(self.recent)[-20:]:
                        f.write(json.dumps(asdict(out)) + '\n')
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def _propagate_learning(self, outcome: Outcome) -> None:
        """Propagate outcome to relevant learning components."""
        try:
            # Update retrieval manager
            if outcome.operation == 'retrieval':
                from jinx.micro.brain.adaptive_retrieval import record_retrieval_outcome
                hits = outcome.metadata.get('hits', 0)
                k = outcome.metadata.get('k', 10)
                query = outcome.metadata.get('query', '')
                useful_ratio = min(1.0, hits / max(1, k))
                
                if query:
                    await record_retrieval_outcome(
                        query=query,
                        k=k,
                        time_ms=int(outcome.latency_ms),
                        hits_count=hits,
                        useful_ratio=useful_ratio
                    )
        except Exception:
            pass
    
    def get_success_rate(self, operation: str) -> float:
        """Get success rate for operation type."""
        metrics = self.metrics.get(operation, {})
        total = metrics.get('total', 0)
        if total == 0:
            return 0.5
        
        if operation == 'code_exec':
            return metrics.get('success', 0) / total
        elif operation == 'retrieval':
            return metrics.get('useful', 0) / total
        elif operation == 'context_build':
            return metrics.get('effective', 0) / total
        elif operation == 'response':
            return metrics.get('quality_sum', 0.0) / total
        
        return 0.5
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        summary = {}
        for op, m in self.metrics.items():
            summary[op] = {
                **m,
                'success_rate': self.get_success_rate(op)
            }
        return summary
    
    async def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in recent outcomes."""
        async with self._lock:
            if len(self.recent) < 10:
                return {}
            
            # Compute rolling success rates
            window_size = 50
            recent_list = list(self.recent)[-window_size:]
            
            by_operation: Dict[str, List[bool]] = {}
            for out in recent_list:
                if out.operation not in by_operation:
                    by_operation[out.operation] = []
                by_operation[out.operation].append(out.success)
            
            rolling_rates = {}
            for op, successes in by_operation.items():
                rolling_rates[f'{op}_rate'] = sum(successes) / len(successes)
            
            # Detect trends
            trends = {}
            for op in by_operation:
                values = [1.0 if s else 0.0 for s in by_operation[op]]
                if len(values) >= 10:
                    first_half = sum(values[:len(values)//2]) / max(1, len(values)//2)
                    second_half = sum(values[len(values)//2:]) / max(1, len(values) - len(values)//2)
                    trend = 'improving' if second_half > first_half + 0.1 else 'stable' if abs(second_half - first_half) < 0.1 else 'degrading'
                    trends[op] = trend
            
            return {
                'rolling_rates': rolling_rates,
                'trends': trends,
                'sample_size': len(recent_list)
            }


# Singleton
_tracker: Optional[OutcomeTracker] = None
_tracker_lock = asyncio.Lock()


async def get_outcome_tracker() -> OutcomeTracker:
    """Get singleton outcome tracker."""
    global _tracker
    if _tracker is None:
        async with _tracker_lock:
            if _tracker is None:
                _tracker = OutcomeTracker()
    return _tracker


async def record_outcome(
    operation: str,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None,
    latency_ms: float = 0.0
) -> None:
    """Record outcome globally."""
    tracker = await get_outcome_tracker()
    await tracker.record(operation, success, metadata, latency_ms)


__all__ = [
    "OutcomeTracker",
    "get_outcome_tracker",
    "record_outcome",
]
