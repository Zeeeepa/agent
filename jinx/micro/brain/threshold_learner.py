"""Self-learning threshold manager using Thompson Sampling and Bayesian optimization.

Replaces hardcoded thresholds with adaptive, data-driven values that learn from outcomes.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ThresholdArm:
    """Bayesian arm for threshold selection using Beta distribution."""
    value: float
    successes: int = 1  # Start with prior
    failures: int = 1   # Start with prior
    
    def sample(self) -> float:
        """Thompson sampling: sample from Beta(successes, failures)."""
        return random.betavariate(self.successes, self.failures)
    
    def update(self, success: bool) -> None:
        """Update with outcome."""
        if success:
            self.successes += 1
        else:
            self.failures += 1
    
    @property
    def mean(self) -> float:
        """Mean of Beta distribution."""
        return self.successes / (self.successes + self.failures)


class ThresholdLearner:
    """Self-learning threshold manager with Bayesian optimization."""
    
    def __init__(self, state_path: str = "log/threshold_learner.json"):
        self.state_path = state_path
        
        # Thresholds for different contexts
        self.thresholds: Dict[str, List[ThresholdArm]] = {
            # Locator classification threshold
            'locator_thresh': [ThresholdArm(v) for v in [0.03, 0.06, 0.10, 0.15, 0.20]],
            
            # Saturation enable/disable ratios
            'saturate_enable': [ThresholdArm(v) for v in [0.6, 0.7, 0.8, 0.85, 0.9]],
            'saturate_disable': [ThresholdArm(v) for v in [0.1, 0.2, 0.3, 0.4, 0.5]],
            
            # Brain activation confidence
            'brain_confidence': [ThresholdArm(v) for v in [0.3, 0.4, 0.5, 0.6, 0.7]],
            
            # Memory usefulness threshold
            'memory_useful': [ThresholdArm(v) for v in [0.3, 0.4, 0.5, 0.6, 0.7]],
            
            # Code similarity threshold
            'code_similarity': [ThresholdArm(v) for v in [0.5, 0.6, 0.7, 0.8, 0.9]],
        }
        
        self._load_state()
        self._lock = asyncio.Lock()
    
    def _load_state(self) -> None:
        """Load persisted state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for key, arms_data in data.get('thresholds', {}).items():
                    if key in self.thresholds:
                        for arm_data in arms_data:
                            for arm in self.thresholds[key]:
                                if abs(arm.value - arm_data['value']) < 0.001:
                                    arm.successes = arm_data.get('successes', 1)
                                    arm.failures = arm_data.get('failures', 1)
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                data = {
                    'thresholds': {
                        key: [{'value': a.value, 'successes': a.successes, 'failures': a.failures} for a in arms]
                        for key, arms in self.thresholds.items()
                    },
                    'timestamp': time.time()
                }
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def select_threshold(self, context: str) -> float:
        """Select threshold using Thompson Sampling."""
        async with self._lock:
            arms = self.thresholds.get(context)
            if not arms:
                return 0.5  # Default fallback
            
            # Thompson Sampling: sample from each arm's posterior
            samples = [(arm, arm.sample()) for arm in arms]
            best_arm, _ = max(samples, key=lambda x: x[1])
            
            return best_arm.value
    
    async def record_outcome(self, context: str, threshold_used: float, success: bool) -> None:
        """Record outcome and update arm."""
        async with self._lock:
            arms = self.thresholds.get(context)
            if not arms:
                return
            
            # Find closest arm
            closest_arm = min(arms, key=lambda a: abs(a.value - threshold_used))
            closest_arm.update(success)
            
            # Periodically save
            if random.random() < 0.1:  # Save 10% of the time
                await self._save_state()
    
    async def get_best_threshold(self, context: str) -> float:
        """Get current best threshold (highest mean)."""
        async with self._lock:
            arms = self.thresholds.get(context)
            if not arms:
                return 0.5
            
            best_arm = max(arms, key=lambda a: a.mean)
            return best_arm.value
    
    def get_stats(self, context: str) -> Dict[str, object]:
        """Get statistics for a threshold context."""
        arms = self.thresholds.get(context, [])
        return {
            str(a.value): {
                'mean': a.mean,
                'successes': a.successes,
                'failures': a.failures,
                'total': a.successes + a.failures
            }
            for a in arms
        }


# Singleton
_learner: Optional[ThresholdLearner] = None
_learner_lock = asyncio.Lock()


async def get_threshold_learner() -> ThresholdLearner:
    """Get singleton threshold learner."""
    global _learner
    if _learner is None:
        async with _learner_lock:
            if _learner is None:
                _learner = ThresholdLearner()
    return _learner


async def select_threshold(context: str) -> float:
    """Select optimal threshold for context."""
    learner = await get_threshold_learner()
    return await learner.select_threshold(context)


async def record_threshold_outcome(context: str, threshold: float, success: bool) -> None:
    """Record threshold outcome."""
    learner = await get_threshold_learner()
    await learner.record_outcome(context, threshold, success)


__all__ = [
    "ThresholdLearner",
    "get_threshold_learner",
    "select_threshold",
    "record_threshold_outcome",
]
