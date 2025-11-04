"""Adaptive Configuration - intelligent real-time configuration with memory.

Self-learning configuration system:
- Uses embeddings to understand system state
- Stores successful configs in memory
- Learns optimal parameters from usage
- Real-time performance monitoring
- Automatic parameter tuning
- Context-aware configuration
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


@dataclass
class ConfigSnapshot:
    """Configuration snapshot with performance metrics."""
    timestamp: float
    config: Dict[str, str]
    
    # Performance metrics
    avg_latency_ms: float
    success_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Context
    workload_type: str  # 'code', 'conversation', 'planning'
    concurrent_tasks: int
    
    # Outcome
    quality_score: float  # 0-1, overall quality
    
    def to_embedding_text(self) -> str:
        """Convert to text for embedding."""
        return (
            f"Config: workload={self.workload_type} "
            f"latency={self.avg_latency_ms:.1f}ms "
            f"success={self.success_rate:.2%} "
            f"memory={self.memory_usage_mb:.0f}MB "
            f"quality={self.quality_score:.2f}"
        )


@dataclass
class PerformanceWindow:
    """Sliding window of performance metrics."""
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    successes: int = 0
    failures: int = 0
    total_requests: int = 0
    
    def add_result(self, latency_ms: float, success: bool):
        """Add result to window."""
        self.latencies.append(latency_ms)
        self.total_requests += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        if not self.latencies:
            return {
                'avg_latency_ms': 0.0,
                'success_rate': 0.0,
                'p95_latency_ms': 0.0
            }
        
        sorted_latencies = sorted(self.latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        
        return {
            'avg_latency_ms': sum(self.latencies) / len(self.latencies),
            'success_rate': self.successes / max(1, self.total_requests),
            'p95_latency_ms': sorted_latencies[p95_idx] if sorted_latencies else 0.0
        }


class AdaptiveConfigManager:
    """Intelligent configuration manager with embeddings and memory."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Current configuration
        self._current_config: Dict[str, str] = {}
        
        # Performance monitoring
        self._performance_window = PerformanceWindow()
        
        # Configuration history (for learning)
        self._config_history: deque[ConfigSnapshot] = deque(maxlen=1000)
        
        # Learning parameters
        self._learning_rate = 0.1
        self._exploration_rate = 0.1  # Epsilon for exploration
        
        # Context tracking
        self._current_workload = 'default'
        self._workload_counters = {
            'code': 0,
            'conversation': 0,
            'planning': 0,
            'default': 0
        }
        
        # Tunable parameters with ranges
        self._tunable_params = {
            'JINX_MAX_CONCURRENT': (1, 10),
            'JINX_FRAME_MAX_CONC': (1, 6),
            'JINX_CHAINED_DIALOG_CTX_MS': (50, 300),
            'JINX_CHAINED_PROJECT_CTX_MS': (200, 1000),
            'EMBED_SLICE_MS': (5, 30),
            'JINX_LOCATOR_VEC_MS': (50, 300),
        }
        
        # Best configs per workload (learned)
        self._best_configs: Dict[str, Dict[str, str]] = {}
        
        # Last tuning time
        self._last_tune_time = time.time()
        self._tune_interval_seconds = 60  # Tune every 60 seconds
    
    async def initialize(self):
        """Initialize with memory retrieval."""
        async with self._lock:
            # Load best configs from memory
            await self._load_from_memory()
            
            # Apply initial optimal config
            await self._apply_optimal_config()
    
    async def record_operation(
        self,
        latency_ms: float,
        success: bool,
        workload_type: str = 'default'
    ):
        """Record operation result for learning."""
        
        async with self._lock:
            # Update performance window
            self._performance_window.add_result(latency_ms, success)
            
            # Update workload tracking
            self._current_workload = workload_type
            self._workload_counters[workload_type] = \
                self._workload_counters.get(workload_type, 0) + 1
            
            # Check if it's time to tune
            current_time = time.time()
            if current_time - self._last_tune_time > self._tune_interval_seconds:
                await self._auto_tune()
                self._last_tune_time = current_time
    
    async def _auto_tune(self):
        """Automatically tune configuration based on performance."""
        
        metrics = self._performance_window.get_metrics()
        
        # Check if performance is suboptimal
        needs_tuning = False
        
        if metrics['avg_latency_ms'] > 500:  # High latency
            needs_tuning = True
        elif metrics['success_rate'] < 0.9:  # Low success rate
            needs_tuning = True
        elif metrics['p95_latency_ms'] > 1000:  # High p95
            needs_tuning = True
        
        if not needs_tuning:
            # Performance is good, save current config
            await self._save_good_config(metrics)
            return
        
        # Performance is suboptimal, try to improve
        await self._optimize_config(metrics)
    
    async def _optimize_config(self, metrics: Dict[str, float]):
        """Optimize configuration based on metrics."""
        
        # Determine optimization strategy
        if metrics['avg_latency_ms'] > 500:
            # High latency - increase concurrency or budgets
            await self._increase_concurrency()
            await self._adjust_budgets(direction='increase')
        
        elif metrics['success_rate'] < 0.9:
            # Low success - decrease concurrency, increase timeouts
            await self._decrease_concurrency()
            await self._adjust_budgets(direction='increase')
        
        # Log optimization
        try:
            from jinx.micro.brain import remember_episode
            await remember_episode(
                content=f"Config optimization: latency={metrics['avg_latency_ms']:.1f}ms, "
                        f"success={metrics['success_rate']:.2%}",
                episode_type='system',
                context={'optimization': 'auto_tune', 'metrics': metrics},
                importance=0.6
            )
        except Exception:
            pass
    
    async def _increase_concurrency(self):
        """Increase concurrency parameters."""
        
        for param, (min_val, max_val) in self._tunable_params.items():
            if 'CONC' in param:
                current = int(os.getenv(param, str((min_val + max_val) // 2)))
                new_value = min(max_val, current + 1)
                os.environ[param] = str(new_value)
                self._current_config[param] = str(new_value)
    
    async def _decrease_concurrency(self):
        """Decrease concurrency parameters."""
        
        for param, (min_val, max_val) in self._tunable_params.items():
            if 'CONC' in param:
                current = int(os.getenv(param, str((min_val + max_val) // 2)))
                new_value = max(min_val, current - 1)
                os.environ[param] = str(new_value)
                self._current_config[param] = str(new_value)
    
    async def _adjust_budgets(self, direction: str = 'increase'):
        """Adjust time budget parameters."""
        
        multiplier = 1.2 if direction == 'increase' else 0.8
        
        for param, (min_val, max_val) in self._tunable_params.items():
            if 'MS' in param or 'CTX' in param:
                current = int(os.getenv(param, str((min_val + max_val) // 2)))
                new_value = int(current * multiplier)
                new_value = max(min_val, min(max_val, new_value))
                os.environ[param] = str(new_value)
                self._current_config[param] = str(new_value)
    
    async def _save_good_config(self, metrics: Dict[str, float]):
        """Save successful configuration to memory."""
        
        # Create snapshot
        snapshot = ConfigSnapshot(
            timestamp=time.time(),
            config=self._current_config.copy(),
            avg_latency_ms=metrics['avg_latency_ms'],
            success_rate=metrics['success_rate'],
            memory_usage_mb=0.0,  # Would get from psutil
            cpu_usage_percent=0.0,
            workload_type=self._current_workload,
            concurrent_tasks=self._workload_counters[self._current_workload],
            quality_score=self._calculate_quality_score(metrics)
        )
        
        # Add to history
        self._config_history.append(snapshot)
        
        # Update best config for this workload
        if self._current_workload not in self._best_configs or \
           snapshot.quality_score > self._get_best_quality(self._current_workload):
            self._best_configs[self._current_workload] = snapshot.config.copy()
            
            # Save to memory with embeddings
            await self._save_to_memory(snapshot)
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        
        # Normalize metrics to 0-1 (higher is better)
        latency_score = max(0, 1.0 - (metrics['avg_latency_ms'] / 1000))
        success_score = metrics['success_rate']
        p95_score = max(0, 1.0 - (metrics['p95_latency_ms'] / 2000))
        
        # Weighted average
        quality = (
            0.4 * latency_score +
            0.4 * success_score +
            0.2 * p95_score
        )
        
        return quality
    
    def _get_best_quality(self, workload: str) -> float:
        """Get quality of best config for workload."""
        
        # Find best quality from history
        best_quality = 0.0
        
        for snapshot in self._config_history:
            if snapshot.workload_type == workload:
                best_quality = max(best_quality, snapshot.quality_score)
        
        return best_quality
    
    async def _save_to_memory(self, snapshot: ConfigSnapshot):
        """Save config snapshot to memory with embeddings."""
        
        try:
            # Save to episodic memory
            from jinx.micro.brain import remember_episode
            
            content = snapshot.to_embedding_text()
            
            await remember_episode(
                content=content,
                episode_type='config_optimization',
                context={
                    'workload': snapshot.workload_type,
                    'quality': snapshot.quality_score,
                    'config': snapshot.config
                },
                importance=0.7
            )
            
            # Save to embedding store for semantic search
            try:
                from .config_embeddings import store_successful_config
                
                await store_successful_config(
                    config=snapshot.config,
                    quality_score=snapshot.quality_score,
                    workload_type=snapshot.workload_type
                )
            except Exception:
                pass
            
            # Also save to knowledge graph
            try:
                from jinx.micro.brain import get_knowledge_graph
                
                kg = await get_knowledge_graph()
                await kg.add_node({
                    'type': 'optimal_config',
                    'workload': snapshot.workload_type,
                    'quality': snapshot.quality_score,
                    'latency': snapshot.avg_latency_ms,
                    'config': snapshot.config
                })
            except Exception:
                pass
        
        except Exception:
            pass
    
    async def _load_from_memory(self):
        """Load best configs from memory."""
        
        try:
            from jinx.micro.brain import search_all_memories
            
            # Search for optimal configs
            memories = await search_all_memories(
                "optimal config high quality",
                k=10
            )
            
            # Extract configs from memories
            for memory in memories:
                if hasattr(memory, 'context'):
                    context = memory.context
                    if 'workload' in context and 'config' in context:
                        workload = context['workload']
                        config = context['config']
                        quality = context.get('quality', 0.0)
                        
                        # Update if better than current
                        if workload not in self._best_configs or \
                           quality > self._get_best_quality(workload):
                            self._best_configs[workload] = config
        
        except Exception:
            pass
    
    async def _apply_optimal_config(self):
        """Apply optimal configuration for current workload."""
        
        workload = self._current_workload
        
        if workload in self._best_configs:
            # Apply best known config
            for key, value in self._best_configs[workload].items():
                os.environ[key] = value
                self._current_config[key] = value
    
    async def get_optimal_for_context(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get optimal configuration for given context using embeddings."""
        
        try:
            # First try semantic search via embedding store
            from .config_embeddings import predict_config_for_context
            
            predicted_config = await predict_config_for_context(context)
            
            if predicted_config:
                return predicted_config
            
            # Fallback to memory search
            from jinx.micro.brain import search_all_memories
            
            # Create context description
            context_text = f"workload={context.get('type', 'default')} "
            
            # Search for similar successful configs
            memories = await search_all_memories(
                f"optimal config {context_text}",
                k=5
            )
            
            # Find best match
            best_config = None
            best_quality = 0.0
            
            for memory in memories:
                if hasattr(memory, 'context'):
                    mem_context = memory.context
                    quality = mem_context.get('quality', 0.0)
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_config = mem_context.get('config', {})
            
            if best_config:
                return best_config
        
        except Exception:
            pass
        
        # Fallback to current config
        return self._current_config
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        
        perf_metrics = self._performance_window.get_metrics()
        
        return {
            'performance': perf_metrics,
            'workload': self._current_workload,
            'workload_counts': self._workload_counters,
            'config_history_size': len(self._config_history),
            'best_configs': {
                workload: self._get_best_quality(workload)
                for workload in self._best_configs
            }
        }


# Singleton
_adaptive_config: Optional[AdaptiveConfigManager] = None
_config_lock = asyncio.Lock()


async def get_adaptive_config() -> AdaptiveConfigManager:
    """Get singleton adaptive config manager."""
    global _adaptive_config
    if _adaptive_config is None:
        async with _config_lock:
            if _adaptive_config is None:
                _adaptive_config = AdaptiveConfigManager()
                await _adaptive_config.initialize()
    return _adaptive_config


async def record_operation_result(
    latency_ms: float,
    success: bool,
    workload_type: str = 'default'
):
    """Record operation result for adaptive learning."""
    config = await get_adaptive_config()
    await config.record_operation(latency_ms, success, workload_type)


__all__ = [
    "AdaptiveConfigManager",
    "ConfigSnapshot",
    "get_adaptive_config",
    "record_operation_result",
]
