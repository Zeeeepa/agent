"""Auto-Tuner - автоматическая непрерывная оптимизация всех систем.

Использует meta-learning для автоматической настройки всех параметров.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ParameterConfig:
    """Configuration for a tunable parameter."""
    system: str
    parameter: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    last_update: float
    performance_history: List[Tuple[float, float]]  # (value, performance)


class AutoTuner:
    """Meta-learning auto-tuner that optimizes all brain systems continuously."""
    
    def __init__(self, state_path: str = "log/auto_tuner.json"):
        self.state_path = state_path
        
        # Registry of tunable parameters
        self.parameters: Dict[str, ParameterConfig] = {}
        
        # Optimization history
        self.optimization_history: deque[Dict[str, Any]] = deque(maxlen=500)
        
        # Learning rate for parameter updates
        self.learning_rate = 0.05
        
        # Exploration strategies
        self.exploration_rate = 0.1
        self.exploration_decay = 0.999
        
        # Performance baselines
        self.baselines: Dict[str, float] = {}
        
        self._lock = asyncio.Lock()
        self._running = False
        self._tuning_task: Optional[asyncio.Task] = None
        
        self._register_tunable_parameters()
        self._load_state()
    
    def _register_tunable_parameters(self) -> None:
        """Register all tunable parameters across all systems."""
        
        # Adaptive Retrieval parameters
        self.parameters['retrieval_exploration'] = ParameterConfig(
            system='adaptive_retrieval',
            parameter='exploration_factor',
            current_value=2.0,
            min_value=0.5,
            max_value=5.0,
            step_size=0.2,
            last_update=time.time(),
            performance_history=[]
        )
        
        # Threshold Learner parameters
        self.parameters['threshold_alpha'] = ParameterConfig(
            system='threshold_learner',
            parameter='alpha',
            current_value=1.0,
            min_value=0.1,
            max_value=5.0,
            step_size=0.1,
            last_update=time.time(),
            performance_history=[]
        )
        
        self.parameters['threshold_beta'] = ParameterConfig(
            system='threshold_learner',
            parameter='beta',
            current_value=1.0,
            min_value=0.1,
            max_value=5.0,
            step_size=0.1,
            last_update=time.time(),
            performance_history=[]
        )
        
        # Context Optimizer parameters
        self.parameters['context_learning_rate'] = ParameterConfig(
            system='context_optimizer',
            parameter='learning_rate',
            current_value=0.1,
            min_value=0.01,
            max_value=0.5,
            step_size=0.02,
            last_update=time.time(),
            performance_history=[]
        )
        
        self.parameters['context_discount'] = ParameterConfig(
            system='context_optimizer',
            parameter='discount_factor',
            current_value=0.95,
            min_value=0.7,
            max_value=0.99,
            step_size=0.02,
            last_update=time.time(),
            performance_history=[]
        )
        
        # Query Classifier parameters
        self.parameters['classifier_alpha'] = ParameterConfig(
            system='query_classifier',
            parameter='alpha',
            current_value=0.1,
            min_value=0.01,
            max_value=0.5,
            step_size=0.02,
            last_update=time.time(),
            performance_history=[]
        )
        
        # Rate Limiter parameters
        self.parameters['ratelimit_increase'] = ParameterConfig(
            system='rate_limiter',
            parameter='increase_factor',
            current_value=1.1,
            min_value=1.05,
            max_value=1.5,
            step_size=0.05,
            last_update=time.time(),
            performance_history=[]
        )
        
        self.parameters['ratelimit_decrease'] = ParameterConfig(
            system='rate_limiter',
            parameter='decrease_factor',
            current_value=0.9,
            min_value=0.5,
            max_value=0.95,
            step_size=0.05,
            last_update=time.time(),
            performance_history=[]
        )
        
        # Predictive Cache parameters
        self.parameters['cache_ttl'] = ParameterConfig(
            system='predictive_cache',
            parameter='ttl_seconds',
            current_value=30.0,
            min_value=10.0,
            max_value=300.0,
            step_size=10.0,
            last_update=time.time(),
            performance_history=[]
        )
        
        # Intelligent Planner parameters
        self.parameters['planner_learning_rate'] = ParameterConfig(
            system='intelligent_planner',
            parameter='learning_rate',
            current_value=0.1,
            min_value=0.01,
            max_value=0.5,
            step_size=0.02,
            last_update=time.time(),
            performance_history=[]
        )
        
        self.parameters['planner_exploration'] = ParameterConfig(
            system='intelligent_planner',
            parameter='exploration_rate',
            current_value=0.15,
            min_value=0.05,
            max_value=0.5,
            step_size=0.05,
            last_update=time.time(),
            performance_history=[]
        )
    
    def _load_state(self) -> None:
        """Load tuner state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore parameters
                for param_name, param_data in data.get('parameters', {}).items():
                    if param_name in self.parameters:
                        self.parameters[param_name].current_value = param_data['current_value']
                        self.parameters[param_name].performance_history = [
                            tuple(h) for h in param_data.get('history', [])
                        ]
                
                # Restore baselines
                self.baselines = data.get('baselines', {})
                
                # Restore exploration rate
                self.exploration_rate = data.get('exploration_rate', 0.1)
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist tuner state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                parameters_data = {}
                for name, config in self.parameters.items():
                    parameters_data[name] = {
                        'current_value': config.current_value,
                        'history': config.performance_history[-20:]  # Last 20
                    }
                
                data = {
                    'parameters': parameters_data,
                    'baselines': self.baselines,
                    'exploration_rate': self.exploration_rate,
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def start(self) -> None:
        """Start automatic tuning."""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._tuning_task = asyncio.create_task(self._tuning_loop())
    
    async def stop(self) -> None:
        """Stop automatic tuning."""
        async with self._lock:
            self._running = False
            
            if self._tuning_task:
                self._tuning_task.cancel()
                try:
                    await self._tuning_task
                except asyncio.CancelledError:
                    pass
    
    async def _tuning_loop(self) -> None:
        """Main tuning loop - runs continuously."""
        while self._running:
            try:
                await asyncio.sleep(45.0)  # Tune every 45 seconds
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(5.0)
                continue
            
            try:
                # Collect current performance metrics
                performance = await self._collect_performance_metrics()
                
                # Tune each parameter
                for param_name, config in list(self.parameters.items()):
                    await self._tune_parameter(config, performance)
                
                # Decay exploration rate
                self.exploration_rate *= self.exploration_decay
                self.exploration_rate = max(0.01, self.exploration_rate)
                
                # Save state periodically
                if len(self.optimization_history) % 5 == 0:
                    await self._save_state()
                
            except Exception:
                pass
    
    async def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect performance metrics from all systems."""
        metrics = {}
        
        try:
            # Outcome Tracker metrics
            from jinx.micro.brain import get_outcome_tracker
            tracker = await get_outcome_tracker()
            all_metrics = tracker.get_all_metrics()
            
            for op, m in all_metrics.items():
                metrics[f"{op}_success_rate"] = m.get('success_rate', 0.5)
            
            # Compute overall performance
            if metrics:
                metrics['overall'] = sum(metrics.values()) / len(metrics)
            else:
                metrics['overall'] = 0.5
        except Exception:
            metrics['overall'] = 0.5
        
        return metrics
    
    async def _tune_parameter(
        self,
        config: ParameterConfig,
        performance: Dict[str, float]
    ) -> None:
        """Tune a single parameter using gradient-based optimization."""
        try:
            current_perf = performance.get('overall', 0.5)
            
            # Record current performance
            config.performance_history.append((config.current_value, current_perf))
            
            # Keep history bounded
            if len(config.performance_history) > 50:
                config.performance_history = config.performance_history[-50:]
            
            # Need at least 3 data points
            if len(config.performance_history) < 3:
                return
            
            # Compute gradient estimate
            gradient = self._estimate_gradient(config)
            
            # Exploration: random perturbation
            explore = (hash(str(time.time())) % 100) < (self.exploration_rate * 100)
            
            if explore:
                # Random step
                import random
                direction = random.choice([-1, 1])
                new_value = config.current_value + direction * config.step_size
            else:
                # Gradient ascent
                new_value = config.current_value + self.learning_rate * gradient * config.step_size
            
            # Clamp to bounds
            new_value = max(config.min_value, min(config.max_value, new_value))
            
            # Apply if different
            if abs(new_value - config.current_value) > 0.001:
                await self._apply_parameter(config, new_value)
                config.current_value = new_value
                config.last_update = time.time()
        
        except Exception:
            pass
    
    def _estimate_gradient(self, config: ParameterConfig) -> float:
        """Estimate gradient from performance history."""
        try:
            history = config.performance_history[-10:]  # Last 10
            
            if len(history) < 3:
                return 0.0
            
            # Simple finite difference
            # gradient ≈ Δperformance / Δvalue
            
            recent = history[-3:]
            
            # Sort by value
            recent.sort(key=lambda x: x[0])
            
            if len(recent) >= 2:
                value_diff = recent[-1][0] - recent[0][0]
                perf_diff = recent[-1][1] - recent[0][1]
                
                if abs(value_diff) > 0.001:
                    gradient = perf_diff / value_diff
                    
                    # Clip gradient
                    return max(-1.0, min(1.0, gradient))
            
            return 0.0
        except Exception:
            return 0.0
    
    async def _apply_parameter(self, config: ParameterConfig, new_value: float) -> None:
        """Apply new parameter value to the system."""
        try:
            # System-specific application logic
            # (In real implementation, would update actual system parameters)
            
            # Record optimization event
            self.optimization_history.append({
                'system': config.system,
                'parameter': config.parameter,
                'old_value': config.current_value,
                'new_value': new_value,
                'timestamp': time.time()
            })
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tuner statistics."""
        param_stats = {}
        for name, config in self.parameters.items():
            param_stats[name] = {
                'current_value': config.current_value,
                'min': config.min_value,
                'max': config.max_value,
                'history_size': len(config.performance_history),
                'last_update': config.last_update
            }
        
        return {
            'parameters': param_stats,
            'exploration_rate': self.exploration_rate,
            'optimizations': len(self.optimization_history),
            'running': self._running
        }


# Singleton
_tuner: Optional[AutoTuner] = None
_tuner_lock = asyncio.Lock()


async def get_auto_tuner() -> AutoTuner:
    """Get singleton auto-tuner."""
    global _tuner
    if _tuner is None:
        async with _tuner_lock:
            if _tuner is None:
                _tuner = AutoTuner()
                # Auto-start
                await _tuner.start()
    return _tuner


__all__ = [
    "AutoTuner",
    "get_auto_tuner",
]
