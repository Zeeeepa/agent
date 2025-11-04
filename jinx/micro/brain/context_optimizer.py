"""Context size optimizer with RL-based budget allocation.

Replaces fixed context sizes with adaptive, outcome-driven optimization.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ContextAllocation:
    """Context budget allocation for a query."""
    total_budget: int  # total chars
    code_budget: int
    brain_budget: int
    memory_budget: int
    refs_budget: int
    confidence: float


@dataclass
class ContextOutcome:
    """Outcome of a context allocation."""
    allocation: ContextAllocation
    used_chars: int
    response_quality: float  # 0-1
    latency_ms: float
    success: bool


class ContextSizeOptimizer:
    """RL-based optimizer for context size allocation."""
    
    def __init__(self, state_path: str = "log/context_optimizer.json"):
        self.state_path = state_path
        
        # Budget ranges for exploration
        self.min_total = 500
        self.max_total = 15000
        self.default_total = 5000
        
        # Component budget ratios (learned)
        self.code_ratio = 0.6
        self.brain_ratio = 0.15
        self.memory_ratio = 0.15
        self.refs_ratio = 0.10
        
        # Outcome history for RL
        self.outcomes: deque[ContextOutcome] = deque(maxlen=200)
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.2
        
        # State-action values: (query_complexity, budget) -> expected_reward
        self.q_values: Dict[Tuple[int, int], float] = {}
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.code_ratio = data.get('code_ratio', self.code_ratio)
                self.brain_ratio = data.get('brain_ratio', self.brain_ratio)
                self.memory_ratio = data.get('memory_ratio', self.memory_ratio)
                self.refs_ratio = data.get('refs_ratio', self.refs_ratio)
                
                # Restore Q-values
                q_data = data.get('q_values', {})
                for key_str, value in q_data.items():
                    complexity, budget = map(int, key_str.split('|'))
                    self.q_values[(complexity, budget)] = value
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize Q-values
                q_data = {f"{k[0]}|{k[1]}": v for k, v in self.q_values.items()}
                
                data = {
                    'code_ratio': self.code_ratio,
                    'brain_ratio': self.brain_ratio,
                    'memory_ratio': self.memory_ratio,
                    'refs_ratio': self.refs_ratio,
                    'q_values': q_data,
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _estimate_query_complexity(self, query: str) -> int:
        """Estimate query complexity (0-10 scale)."""
        if not query:
            return 3
        
        q = query.strip()
        
        # Length factor
        len_score = min(5, len(q) / 100)
        
        # Structural complexity
        code_chars = sum(1 for c in q if c in '(){}[]<>=+-*/|&%^~')
        struct_score = min(3, code_chars / 20)
        
        # Line count
        line_score = min(2, len(q.split('\n')) / 5)
        
        complexity = int(len_score + struct_score + line_score)
        return min(10, max(0, complexity))
    
    def _discretize_budget(self, budget: int) -> int:
        """Discretize budget into bins for Q-learning."""
        if budget < 1000:
            return 0
        elif budget < 3000:
            return 1
        elif budget < 6000:
            return 2
        elif budget < 10000:
            return 3
        else:
            return 4
    
    def _get_q_value(self, complexity: int, budget_bin: int) -> float:
        """Get Q-value for state-action pair."""
        return self.q_values.get((complexity, budget_bin), 0.0)
    
    def _set_q_value(self, complexity: int, budget_bin: int, value: float) -> None:
        """Set Q-value for state-action pair."""
        self.q_values[(complexity, budget_bin)] = value
    
    async def allocate(self, query: str) -> ContextAllocation:
        """Allocate context budget for query."""
        async with self._lock:
            complexity = self._estimate_query_complexity(query)
            
            # Exploration vs exploitation
            if len(self.outcomes) < 20 or (len(self.outcomes) > 0 and self.outcomes[-1].response_quality < 0.5):
                # Increase exploration when learning or after poor outcomes
                explore = True
            else:
                explore = (hash(query) % 100) < (self.exploration_rate * 100)
            
            if explore:
                # Explore: random budget
                total_budget = int(self.min_total + (self.max_total - self.min_total) * (hash(query) % 100) / 100)
            else:
                # Exploit: use best known budget for this complexity
                best_budget_bin = -1
                best_q = float('-inf')
                
                for budget_bin in range(5):
                    q = self._get_q_value(complexity, budget_bin)
                    if q > best_q:
                        best_q = q
                        best_budget_bin = budget_bin
                
                # Map bin to actual budget
                budget_map = [750, 2000, 4500, 8000, 12000]
                total_budget = budget_map[best_budget_bin] if best_budget_bin >= 0 else self.default_total
            
            # Allocate components
            code_budget = int(total_budget * self.code_ratio)
            brain_budget = int(total_budget * self.brain_ratio)
            memory_budget = int(total_budget * self.memory_ratio)
            refs_budget = int(total_budget * self.refs_ratio)
            
            # Confidence based on Q-value
            budget_bin = self._discretize_budget(total_budget)
            q_val = self._get_q_value(complexity, budget_bin)
            confidence = 1.0 / (1.0 + math.exp(-q_val))  # Sigmoid
            
            return ContextAllocation(
                total_budget=total_budget,
                code_budget=code_budget,
                brain_budget=brain_budget,
                memory_budget=memory_budget,
                refs_budget=refs_budget,
                confidence=confidence
            )
    
    async def record_outcome(
        self,
        query: str,
        allocation: ContextAllocation,
        used_chars: int,
        response_quality: float,
        latency_ms: float,
        success: bool
    ) -> None:
        """Record outcome and update Q-values."""
        async with self._lock:
            outcome = ContextOutcome(
                allocation=allocation,
                used_chars=used_chars,
                response_quality=response_quality,
                latency_ms=latency_ms,
                success=success
            )
            
            self.outcomes.append(outcome)
            
            # Compute reward
            # Reward = quality - efficiency_penalty - latency_penalty
            efficiency = used_chars / max(1, allocation.total_budget)
            efficiency_penalty = 0.2 if efficiency < 0.3 else 0.0  # Penalize waste
            latency_penalty = min(0.3, latency_ms / 10000)
            
            reward = response_quality - efficiency_penalty - latency_penalty
            
            # Q-learning update
            complexity = self._estimate_query_complexity(query)
            budget_bin = self._discretize_budget(allocation.total_budget)
            
            current_q = self._get_q_value(complexity, budget_bin)
            
            # Estimate max future Q (assume next state is similar complexity)
            max_future_q = max(
                [self._get_q_value(complexity, b) for b in range(5)],
                default=0.0
            )
            
            # Q-learning update rule
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_future_q - current_q
            )
            
            self._set_q_value(complexity, budget_bin, new_q)
            
            # Update component ratios using gradient descent
            if success and len(self.outcomes) >= 10:
                recent = list(self.outcomes)[-10:]
                
                # Calculate which component had most impact
                # (simplified heuristic: assume code is most important if quality high)
                avg_quality = sum(o.response_quality for o in recent) / len(recent)
                
                if avg_quality > 0.7:
                    # Good outcomes -> slightly increase successful ratios
                    pass  # Keep current ratios stable
                else:
                    # Poor outcomes -> adjust
                    # Increase code ratio slightly
                    self.code_ratio = min(0.75, self.code_ratio + 0.01)
                    # Normalize
                    total = self.code_ratio + self.brain_ratio + self.memory_ratio + self.refs_ratio
                    if total > 0:
                        self.code_ratio /= total
                        self.brain_ratio /= total
                        self.memory_ratio /= total
                        self.refs_ratio /= total
            
            # Periodically save
            if len(self.outcomes) % 10 == 0:
                await self._save_state()
    
    def get_stats(self) -> Dict[str, object]:
        """Get optimizer statistics."""
        if not self.outcomes:
            return {
                'outcomes_count': 0,
                'avg_quality': 0.0,
                'ratios': {
                    'code': self.code_ratio,
                    'brain': self.brain_ratio,
                    'memory': self.memory_ratio,
                    'refs': self.refs_ratio
                }
            }
        
        avg_quality = sum(o.response_quality for o in self.outcomes) / len(self.outcomes)
        avg_efficiency = sum(o.used_chars / max(1, o.allocation.total_budget) for o in self.outcomes) / len(self.outcomes)
        
        return {
            'outcomes_count': len(self.outcomes),
            'avg_quality': avg_quality,
            'avg_efficiency': avg_efficiency,
            'ratios': {
                'code': self.code_ratio,
                'brain': self.brain_ratio,
                'memory': self.memory_ratio,
                'refs': self.refs_ratio
            },
            'q_values_learned': len(self.q_values)
        }


# Singleton
_optimizer: Optional[ContextSizeOptimizer] = None
_optimizer_lock = asyncio.Lock()


async def get_context_optimizer() -> ContextSizeOptimizer:
    """Get singleton context optimizer."""
    global _optimizer
    if _optimizer is None:
        async with _optimizer_lock:
            if _optimizer is None:
                _optimizer = ContextSizeOptimizer()
    return _optimizer


async def allocate_context_budget(query: str) -> ContextAllocation:
    """Allocate optimal context budget for query."""
    optimizer = await get_context_optimizer()
    return await optimizer.allocate(query)


async def record_context_outcome(
    query: str,
    allocation: ContextAllocation,
    used_chars: int,
    response_quality: float,
    latency_ms: float,
    success: bool
) -> None:
    """Record context allocation outcome."""
    optimizer = await get_context_optimizer()
    await optimizer.record_outcome(query, allocation, used_chars, response_quality, latency_ms, success)


__all__ = [
    "ContextAllocation",
    "ContextSizeOptimizer",
    "get_context_optimizer",
    "allocate_context_budget",
    "record_context_outcome",
]
