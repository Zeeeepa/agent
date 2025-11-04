"""Adaptive retrieval system with ML-based parameter tuning and self-learning.

This module replaces static retrieval parameters with intelligent, self-optimizing strategies:
- Multi-armed bandit for k selection
- Reinforcement learning for timeout budgets
- Hit-rate tracking and adaptation
- Query complexity classification
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

try:
    from jinx.log_paths import BLUE_WHISPERS
    from jinx.logger.file_logger import append_line as _log
except Exception:
    _log = None  # type: ignore
    BLUE_WHISPERS = ""


@dataclass
class RetrievalOutcome:
    """Record of a retrieval attempt and its outcome."""
    query: str
    k: int
    time_ms: int
    hits_count: int
    useful_ratio: float  # 0-1, estimated by downstream usage
    timestamp: float
    query_complexity: float  # 0-1, estimated complexity


@dataclass
class BanditArm:
    """Multi-armed bandit arm for parameter selection."""
    value: int  # parameter value (e.g., k=5, k=10, k=20)
    pulls: int = 0
    total_reward: float = 0.0
    
    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.pulls)
    
    def ucb1(self, total_pulls: int, exploration: float = 2.0) -> float:
        """Upper Confidence Bound 1 score."""
        if self.pulls == 0:
            return float('inf')
        exploit = self.mean_reward
        explore = math.sqrt(exploration * math.log(total_pulls) / self.pulls)
        return exploit + explore


class AdaptiveRetrievalManager:
    """Self-learning retrieval parameter manager using bandits and RL."""
    
    def __init__(self, state_path: str = "log/adaptive_retrieval.json"):
        self.state_path = state_path
        
        # Multi-armed bandits for different parameters
        self.k_arms: List[BanditArm] = [
            BanditArm(value=v) for v in [5, 10, 15, 20, 30]
        ]
        self.timeout_arms: List[BanditArm] = [
            BanditArm(value=v) for v in [100, 200, 300, 500, 800]
        ]
        
        # Outcome history for learning
        self.history: deque[RetrievalOutcome] = deque(maxlen=200)
        
        # Query complexity classifier (simple heuristics, can be ML later)
        self.complexity_weights = {
            'length': 0.2,
            'tokens': 0.3,
            'code_like': 0.3,
            'specificity': 0.2
        }
        
        # Adaptive thresholds
        self.min_useful_ratio = 0.3
        self.target_useful_ratio = 0.7
        
        self._load_state()
        self._lock = asyncio.Lock()
    
    def _load_state(self) -> None:
        """Load persisted state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore arms
                for arm_data in data.get('k_arms', []):
                    for arm in self.k_arms:
                        if arm.value == arm_data['value']:
                            arm.pulls = arm_data.get('pulls', 0)
                            arm.total_reward = arm_data.get('total_reward', 0.0)
                
                for arm_data in data.get('timeout_arms', []):
                    for arm in self.timeout_arms:
                        if arm.value == arm_data['value']:
                            arm.pulls = arm_data.get('pulls', 0)
                            arm.total_reward = arm_data.get('total_reward', 0.0)
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist state asynchronously."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                data = {
                    'k_arms': [{'value': a.value, 'pulls': a.pulls, 'total_reward': a.total_reward} for a in self.k_arms],
                    'timeout_arms': [{'value': a.value, 'pulls': a.pulls, 'total_reward': a.total_reward} for a in self.timeout_arms],
                    'timestamp': time.time()
                }
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1) using heuristics.
        
        Higher complexity -> need more thorough search.
        """
        if not query:
            return 0.5
        
        q = query.strip()
        
        # Length factor
        length_score = min(1.0, len(q) / 200.0)
        
        # Token count
        tokens = q.split()
        token_score = min(1.0, len(tokens) / 15.0)
        
        # Code-like detection (symbols, operators)
        code_chars = sum(1 for c in q if c in '(){}[].=<>!&|')
        code_score = min(1.0, code_chars / max(1, len(q)) * 2.0)
        
        # Specificity (lowercase ratio, proper names)
        alpha_chars = [c for c in q if c.isalpha()]
        specificity_score = 0.5
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            specificity_score = min(1.0, upper_ratio * 2.0)
        
        # Weighted combination
        complexity = (
            self.complexity_weights['length'] * length_score +
            self.complexity_weights['tokens'] * token_score +
            self.complexity_weights['code_like'] * code_score +
            self.complexity_weights['specificity'] * specificity_score
        )
        
        return float(max(0.0, min(1.0, complexity)))
    
    async def select_k(self, query: str) -> int:
        """Select optimal k using UCB1 bandit."""
        async with self._lock:
            complexity = self.estimate_query_complexity(query)
            
            # Higher complexity -> bias toward larger k
            total_pulls = sum(arm.pulls for arm in self.k_arms)
            
            # Adjust exploration based on complexity
            exploration_factor = 1.5 + complexity
            
            # Select arm with highest UCB1
            best_arm = max(self.k_arms, key=lambda a: a.ucb1(total_pulls, exploration_factor))
            
            return best_arm.value
    
    async def select_timeout(self, query: str, k: int) -> int:
        """Select optimal timeout using UCB1 bandit."""
        async with self._lock:
            complexity = self.estimate_query_complexity(query)
            
            total_pulls = sum(arm.pulls for arm in self.timeout_arms)
            
            # Higher k and complexity -> bias toward longer timeouts
            k_factor = math.log(k + 1) / math.log(31)  # normalized to 0-1
            exploration_factor = 1.5 + (complexity + k_factor) / 2.0
            
            best_arm = max(self.timeout_arms, key=lambda a: a.ucb1(total_pulls, exploration_factor))
            
            return best_arm.value
    
    async def record_outcome(
        self,
        query: str,
        k: int,
        time_ms: int,
        hits_count: int,
        useful_ratio: float
    ) -> None:
        """Record outcome and update bandit rewards."""
        async with self._lock:
            complexity = self.estimate_query_complexity(query)
            
            outcome = RetrievalOutcome(
                query=query,
                k=k,
                time_ms=time_ms,
                hits_count=hits_count,
                useful_ratio=useful_ratio,
                timestamp=time.time(),
                query_complexity=complexity
            )
            
            self.history.append(outcome)
            
            # Compute reward: balance useful_ratio, efficiency, and hit count
            # Reward = useful_ratio * hit_quality - time_penalty
            hit_quality = min(1.0, hits_count / max(1, k))
            time_penalty = min(0.3, time_ms / 1000.0 * 0.1)
            reward = useful_ratio * hit_quality - time_penalty
            
            # Update k arm
            for arm in self.k_arms:
                if arm.value == k:
                    arm.pulls += 1
                    arm.total_reward += reward
                    break
            
            # Update timeout arm (find closest)
            timeout_arm = min(self.timeout_arms, key=lambda a: abs(a.value - time_ms))
            timeout_arm.pulls += 1
            timeout_arm.total_reward += reward
            
            # Periodically save state
            if len(self.history) % 10 == 0:
                await self._save_state()
            
            # Log significant insights
            try:
                if _log and BLUE_WHISPERS:
                    if len(self.history) % 20 == 0:
                        best_k = max(self.k_arms, key=lambda a: a.mean_reward)
                        best_timeout = max(self.timeout_arms, key=lambda a: a.mean_reward)
                        await _log(BLUE_WHISPERS, f"[adaptive_retrieval] best_k={best_k.value} (reward={best_k.mean_reward:.3f}), best_timeout={best_timeout.value}ms")
            except Exception:
                pass
    
    async def get_current_best(self) -> Tuple[int, int]:
        """Get current best k and timeout based on mean rewards."""
        async with self._lock:
            best_k_arm = max(self.k_arms, key=lambda a: a.mean_reward if a.pulls > 0 else 0.0)
            best_timeout_arm = max(self.timeout_arms, key=lambda a: a.mean_reward if a.pulls > 0 else 0.0)
            
            return best_k_arm.value, best_timeout_arm.value
    
    def get_stats(self) -> Dict[str, object]:
        """Get current statistics."""
        k_stats = {str(a.value): {'pulls': a.pulls, 'reward': a.mean_reward} for a in self.k_arms if a.pulls > 0}
        timeout_stats = {str(a.value): {'pulls': a.pulls, 'reward': a.mean_reward} for a in self.timeout_arms if a.pulls > 0}
        
        avg_useful = sum(o.useful_ratio for o in self.history) / max(1, len(self.history))
        avg_complexity = sum(o.query_complexity for o in self.history) / max(1, len(self.history))
        
        return {
            'k_arms': k_stats,
            'timeout_arms': timeout_stats,
            'outcomes_count': len(self.history),
            'avg_useful_ratio': avg_useful,
            'avg_query_complexity': avg_complexity
        }


# Singleton instance
_manager: Optional[AdaptiveRetrievalManager] = None
_manager_lock = asyncio.Lock()


async def get_adaptive_manager() -> AdaptiveRetrievalManager:
    """Get singleton adaptive retrieval manager."""
    global _manager
    if _manager is None:
        async with _manager_lock:
            if _manager is None:
                _manager = AdaptiveRetrievalManager()
    return _manager


async def select_retrieval_params(query: str) -> Tuple[int, int]:
    """Select optimal k and timeout for query using adaptive learning.
    
    Returns:
        (k, timeout_ms) tuple
    """
    mgr = await get_adaptive_manager()
    k = await mgr.select_k(query)
    timeout = await mgr.select_timeout(query, k)
    return k, timeout


async def record_retrieval_outcome(
    query: str,
    k: int,
    time_ms: int,
    hits_count: int,
    useful_ratio: float = 0.5
) -> None:
    """Record retrieval outcome for learning."""
    mgr = await get_adaptive_manager()
    await mgr.record_outcome(query, k, time_ms, hits_count, useful_ratio)


__all__ = [
    "AdaptiveRetrievalManager",
    "get_adaptive_manager",
    "select_retrieval_params",
    "record_retrieval_outcome",
]
