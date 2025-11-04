"""Intelligent planning system with RL-based task decomposition.

Replaces primitive sequential planning with ML-driven adaptive strategies.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class PlanStep:
    """Intelligent plan step with learned properties."""
    description: str
    estimated_complexity: float  # 0-1
    dependencies: List[int]  # Step indices
    priority: int
    confidence: float
    alternative_approaches: List[str]


@dataclass
class ExecutionPlan:
    """Complete execution plan with metadata."""
    goal: str
    steps: List[PlanStep]
    total_estimated_time: float
    confidence: float
    strategy: str  # 'sequential', 'parallel', 'adaptive'


class IntelligentPlanner:
    """RL-based adaptive planning system."""
    
    def __init__(self, state_path: str = "log/intelligent_planner.json"):
        self.state_path = state_path
        
        # Planning history for learning
        self.history: deque[Tuple[str, str, bool, float]] = deque(maxlen=200)  # (goal, strategy, success, time)
        
        # Strategy performance (learned)
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'uses': 0,
            'successes': 0,
            'avg_time': 0.0
        })
        
        # Complexity estimation model (simple heuristics for now, can be ML later)
        self.complexity_features: Dict[str, float] = {
            'file_ops': 1.5,
            'network': 2.0,
            'ml_inference': 3.0,
            'code_gen': 2.5,
            'refactor': 2.0,
            'search': 1.0,
            'explain': 0.8,
        }
        
        # Q-values for strategy selection: (goal_complexity, strategy) -> expected_reward
        self.q_values: Dict[Tuple[int, str], float] = {}
        
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.15
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted planning state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore strategy performance
                for strategy, perf in data.get('strategy_performance', {}).items():
                    self.strategy_performance[strategy] = perf
                
                # Restore Q-values
                q_data = data.get('q_values', {})
                for key_str, value in q_data.items():
                    complexity, strategy = key_str.split('|')
                    self.q_values[(int(complexity), strategy)] = value
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist planning state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize Q-values
                q_data = {f"{k[0]}|{k[1]}": v for k, v in self.q_values.items()}
                
                data = {
                    'strategy_performance': dict(self.strategy_performance),
                    'q_values': q_data,
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _estimate_goal_complexity(self, goal: str) -> int:
        """Estimate goal complexity (0-10 scale)."""
        g = goal.lower()
        
        complexity = 3  # Base
        
        # Feature-based complexity
        for feature, weight in self.complexity_features.items():
            if feature in g:
                complexity += weight
        
        # Length factor
        complexity += min(3, len(goal) / 100)
        
        # Multiple steps indicator
        if any(word in g for word in ['and', 'then', 'after', 'also', 'plus']):
            complexity += 1.5
        
        # Uncertainty indicators
        if any(word in g for word in ['maybe', 'possibly', 'might', 'could']):
            complexity += 1.0
        
        return int(min(10, max(0, complexity)))
    
    def _select_strategy(self, goal_complexity: int, explore: bool = True) -> str:
        """Select planning strategy using RL (Q-learning)."""
        strategies = ['sequential', 'parallel', 'adaptive']
        
        # Exploration vs exploitation
        if explore and (hash(str(time.time())) % 100) < (self.exploration_rate * 100):
            # Explore: random strategy
            import random
            return random.choice(strategies)
        
        # Exploit: use best Q-value
        best_strategy = 'sequential'
        best_q = float('-inf')
        
        for strategy in strategies:
            q = self.q_values.get((goal_complexity, strategy), 0.0)
            if q > best_q:
                best_q = q
                best_strategy = strategy
        
        return best_strategy
    
    async def plan(self, goal: str, context: Optional[Dict] = None) -> ExecutionPlan:
        """Create intelligent execution plan."""
        async with self._lock:
            g = goal.strip()
            if not g:
                return ExecutionPlan(g, [], 0.0, 0.3, 'sequential')
            
            # Estimate complexity
            complexity = self._estimate_goal_complexity(g)
            
            # Select strategy
            strategy = self._select_strategy(complexity)
            
            # Decompose goal into steps (intelligent decomposition)
            steps = await self._decompose_goal(g, strategy, complexity, context)
            
            # Estimate total time
            total_time = sum(
                step.estimated_complexity * 5.0  # ~5 seconds per complexity unit
                for step in steps
            )
            
            # Compute confidence
            confidence = self._compute_plan_confidence(steps, strategy, complexity)
            
            return ExecutionPlan(
                goal=g,
                steps=steps,
                total_estimated_time=total_time,
                confidence=confidence,
                strategy=strategy
            )
    
    async def _decompose_goal(
        self,
        goal: str,
        strategy: str,
        complexity: int,
        context: Optional[Dict]
    ) -> List[PlanStep]:
        """Decompose goal into executable steps using intelligent heuristics."""
        steps: List[PlanStep] = []
        
        g_lower = goal.lower()
        
        # Pattern-based decomposition (can be replaced with ML model)
        
        # Code execution pattern
        if any(word in g_lower for word in ['create', 'write', 'implement', 'build']):
            steps.append(PlanStep(
                description="Analyze requirements and context",
                estimated_complexity=0.3,
                dependencies=[],
                priority=1,
                confidence=0.9,
                alternative_approaches=["Skip analysis for simple tasks"]
            ))
            
            steps.append(PlanStep(
                description="Generate code implementation",
                estimated_complexity=0.7,
                dependencies=[0],
                priority=2,
                confidence=0.8,
                alternative_approaches=["Use template", "Adapt existing code"]
            ))
        
        # File operations pattern
        if any(word in g_lower for word in ['file', 'read', 'write', 'save']):
            steps.append(PlanStep(
                description="Validate file paths and permissions",
                estimated_complexity=0.2,
                dependencies=[],
                priority=1,
                confidence=0.95,
                alternative_approaches=[]
            ))
            
            steps.append(PlanStep(
                description="Execute file operation",
                estimated_complexity=0.4,
                dependencies=[0],
                priority=2,
                confidence=0.85,
                alternative_approaches=["Async I/O", "Buffered I/O"]
            ))
        
        # Refactoring pattern
        if any(word in g_lower for word in ['refactor', 'improve', 'optimize', 'clean']):
            steps.append(PlanStep(
                description="Analyze current code structure",
                estimated_complexity=0.5,
                dependencies=[],
                priority=1,
                confidence=0.85,
                alternative_approaches=["AST analysis", "Pattern matching"]
            ))
            
            steps.append(PlanStep(
                description="Identify improvement opportunities",
                estimated_complexity=0.6,
                dependencies=[0],
                priority=2,
                confidence=0.75,
                alternative_approaches=["Static analysis", "ML-based detection"]
            ))
            
            steps.append(PlanStep(
                description="Apply refactoring transformations",
                estimated_complexity=0.7,
                dependencies=[1],
                priority=3,
                confidence=0.7,
                alternative_approaches=["Manual edit", "Automated tools"]
            ))
        
        # Generic fallback
        if not steps:
            steps.append(PlanStep(
                description="Understand the goal",
                estimated_complexity=0.3,
                dependencies=[],
                priority=1,
                confidence=0.7,
                alternative_approaches=[]
            ))
            
            steps.append(PlanStep(
                description="Execute the task",
                estimated_complexity=0.6,
                dependencies=[0],
                priority=2,
                confidence=0.6,
                alternative_approaches=["Break into subtasks"]
            ))
        
        # Strategy-specific adjustments
        if strategy == 'parallel':
            # Remove dependencies where possible
            for step in steps[1:]:
                if len(step.dependencies) == 1:
                    step.dependencies = []
        
        elif strategy == 'adaptive':
            # Add checkpoints
            if len(steps) > 2:
                mid = len(steps) // 2
                steps.insert(mid, PlanStep(
                    description="Checkpoint: Validate progress",
                    estimated_complexity=0.1,
                    dependencies=list(range(mid)),
                    priority=mid + 1,
                    confidence=0.9,
                    alternative_approaches=[]
                ))
        
        return steps
    
    def _compute_plan_confidence(
        self,
        steps: List[PlanStep],
        strategy: str,
        complexity: int
    ) -> float:
        """Compute overall plan confidence."""
        if not steps:
            return 0.3
        
        # Average step confidence
        avg_confidence = sum(s.confidence for s in steps) / len(steps)
        
        # Strategy performance factor
        perf = self.strategy_performance.get(strategy, {})
        if perf.get('uses', 0) > 0:
            success_rate = perf.get('successes', 0) / perf['uses']
            avg_confidence *= (0.7 + success_rate * 0.3)
        
        # Complexity penalty
        complexity_penalty = 1.0 - (complexity / 20.0)
        
        return min(1.0, avg_confidence * complexity_penalty)
    
    async def record_execution(
        self,
        goal: str,
        strategy: str,
        success: bool,
        elapsed_time: float
    ) -> None:
        """Record plan execution outcome for learning."""
        async with self._lock:
            complexity = self._estimate_goal_complexity(goal)
            
            # Update history
            self.history.append((goal, strategy, success, elapsed_time))
            
            # Update strategy performance
            perf = self.strategy_performance[strategy]
            perf['uses'] += 1
            if success:
                perf['successes'] += 1
            
            # Update average time (EMA)
            alpha = 0.2
            current_avg = perf.get('avg_time', 0.0)
            perf['avg_time'] = alpha * elapsed_time + (1 - alpha) * current_avg
            
            # Q-learning update
            reward = 1.0 if success else -0.5
            
            # Time penalty
            expected_time = 10.0 + complexity * 2.0
            if elapsed_time > expected_time:
                reward -= min(0.5, (elapsed_time - expected_time) / expected_time)
            
            # Get current Q-value
            state = (complexity, strategy)
            current_q = self.q_values.get(state, 0.0)
            
            # Estimate max future Q
            strategies = ['sequential', 'parallel', 'adaptive']
            max_future_q = max(
                [self.q_values.get((complexity, s), 0.0) for s in strategies],
                default=0.0
            )
            
            # Q-learning update
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_future_q - current_q
            )
            
            self.q_values[state] = new_q
            
            # Periodically save
            if len(self.history) % 10 == 0:
                await self._save_state()
    
    def get_stats(self) -> Dict[str, object]:
        """Get planner statistics."""
        return {
            'total_plans': len(self.history),
            'strategy_performance': dict(self.strategy_performance),
            'q_values_learned': len(self.q_values)
        }


# Singleton
_planner: Optional[IntelligentPlanner] = None
_planner_lock = asyncio.Lock()


async def get_intelligent_planner() -> IntelligentPlanner:
    """Get singleton intelligent planner."""
    global _planner
    if _planner is None:
        async with _planner_lock:
            if _planner is None:
                _planner = IntelligentPlanner()
    return _planner


async def create_intelligent_plan(goal: str, context: Optional[Dict] = None) -> ExecutionPlan:
    """Create intelligent execution plan."""
    planner = await get_intelligent_planner()
    return await planner.plan(goal, context)


async def record_plan_execution(goal: str, strategy: str, success: bool, elapsed_time: float) -> None:
    """Record plan execution outcome."""
    planner = await get_intelligent_planner()
    await planner.record_execution(goal, strategy, success, elapsed_time)


__all__ = [
    "IntelligentPlanner",
    "ExecutionPlan",
    "PlanStep",
    "get_intelligent_planner",
    "create_intelligent_plan",
    "record_plan_execution",
]
