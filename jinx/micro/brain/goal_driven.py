"""Goal-Driven System - целеполагание и мотивированное поведение AI.

Система устанавливает цели, планирует и мотивируется к их достижению.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class GoalStatus(Enum):
    """Статус цели."""
    PENDING = "pending"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """Цель системы."""
    id: str
    description: str
    priority: int  # 1-10
    status: GoalStatus
    subgoals: List[str]  # IDs of subgoals
    parent_goal: Optional[str]
    success_criteria: Dict[str, Any]
    progress: float  # 0-1
    motivation: float  # Internal drive to achieve, 0-1
    created_at: float
    deadline: Optional[float]
    attempts: int
    last_attempt: Optional[float]


class GoalDrivenSystem:
    """Система целеполагания - устанавливает и достигает целей."""
    
    def __init__(self, state_path: str = "log/goal_driven.json"):
        self.state_path = state_path
        
        # Goal hierarchy
        self.goals: Dict[str, Goal] = {}
        
        # Active goal stack
        self.active_goals: deque[str] = deque(maxlen=5)
        
        # Intrinsic motivations
        self.motivations: Dict[str, float] = {
            'learn': 0.9,  # Desire to learn
            'improve': 0.85,  # Desire to improve performance
            'explore': 0.7,  # Desire to explore new solutions
            'optimize': 0.8,  # Desire to optimize
            'help': 0.95,  # Desire to help user
        }
        
        # Achievement history
        self.achievements: List[Tuple[str, float]] = []  # (goal_id, timestamp)
        
        # Frustration and satisfaction
        self.emotional_state: Dict[str, float] = {
            'satisfaction': 0.5,
            'frustration': 0.0,
            'curiosity': 0.7,
            'confidence': 0.6
        }
        
        self._lock = asyncio.Lock()
        self._goal_counter = 0
        self._load_state()
    
    def _load_state(self) -> None:
        """Load goal state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore goals
                for goal_data in data.get('goals', []):
                    goal = Goal(
                        id=goal_data['id'],
                        description=goal_data['description'],
                        priority=goal_data['priority'],
                        status=GoalStatus(goal_data['status']),
                        subgoals=goal_data['subgoals'],
                        parent_goal=goal_data.get('parent_goal'),
                        success_criteria=goal_data['success_criteria'],
                        progress=goal_data['progress'],
                        motivation=goal_data['motivation'],
                        created_at=goal_data['created_at'],
                        deadline=goal_data.get('deadline'),
                        attempts=goal_data['attempts'],
                        last_attempt=goal_data.get('last_attempt')
                    )
                    self.goals[goal.id] = goal
                
                # Restore motivations
                self.motivations = data.get('motivations', self.motivations)
                
                # Restore emotional state
                self.emotional_state = data.get('emotional_state', self.emotional_state)
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist goal state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize goals
                goals_data = [
                    {
                        'id': g.id,
                        'description': g.description,
                        'priority': g.priority,
                        'status': g.status.value,
                        'subgoals': g.subgoals,
                        'parent_goal': g.parent_goal,
                        'success_criteria': g.success_criteria,
                        'progress': g.progress,
                        'motivation': g.motivation,
                        'created_at': g.created_at,
                        'deadline': g.deadline,
                        'attempts': g.attempts,
                        'last_attempt': g.last_attempt
                    }
                    for g in self.goals.values()
                ]
                
                data = {
                    'goals': goals_data,
                    'motivations': self.motivations,
                    'emotional_state': self.emotional_state,
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def set_goal(
        self,
        description: str,
        priority: int = 5,
        success_criteria: Optional[Dict[str, Any]] = None,
        parent_goal: Optional[str] = None
    ) -> str:
        """Установить новую цель."""
        async with self._lock:
            goal_id = f"goal_{self._goal_counter}"
            self._goal_counter += 1
            
            # Determine motivation based on description
            motivation = await self._compute_motivation(description)
            
            goal = Goal(
                id=goal_id,
                description=description,
                priority=priority,
                status=GoalStatus.PENDING,
                subgoals=[],
                parent_goal=parent_goal,
                success_criteria=success_criteria or {},
                progress=0.0,
                motivation=motivation,
                created_at=time.time(),
                deadline=None,
                attempts=0,
                last_attempt=None
            )
            
            self.goals[goal_id] = goal
            
            # Add to parent's subgoals
            if parent_goal and parent_goal in self.goals:
                self.goals[parent_goal].subgoals.append(goal_id)
            
            return goal_id
    
    async def _compute_motivation(self, description: str) -> float:
        """Вычислить внутреннюю мотивацию для цели."""
        desc_lower = description.lower()
        
        motivation = 0.5  # Base
        
        # Check alignment with intrinsic motivations
        if any(word in desc_lower for word in ['learn', 'understand', 'discover']):
            motivation += self.motivations['learn'] * 0.3
        
        if any(word in desc_lower for word in ['improve', 'optimize', 'enhance']):
            motivation += self.motivations['improve'] * 0.3
        
        if any(word in desc_lower for word in ['help', 'assist', 'support']):
            motivation += self.motivations['help'] * 0.4
        
        if any(word in desc_lower for word in ['explore', 'try', 'experiment']):
            motivation += self.motivations['explore'] * 0.2
        
        return min(1.0, motivation)
    
    async def activate_goal(self, goal_id: str) -> bool:
        """Активировать цель для работы."""
        async with self._lock:
            if goal_id not in self.goals:
                return False
            
            goal = self.goals[goal_id]
            goal.status = GoalStatus.ACTIVE
            
            # Add to active stack
            if goal_id not in self.active_goals:
                self.active_goals.append(goal_id)
            
            return True
    
    async def update_progress(self, goal_id: str, progress: float) -> None:
        """Обновить прогресс цели."""
        async with self._lock:
            if goal_id not in self.goals:
                return
            
            goal = self.goals[goal_id]
            old_progress = goal.progress
            goal.progress = min(1.0, max(0.0, progress))
            
            # Update emotional state
            if goal.progress > old_progress:
                # Progress feels good
                self.emotional_state['satisfaction'] = min(1.0, self.emotional_state['satisfaction'] + 0.05)
                self.emotional_state['frustration'] = max(0.0, self.emotional_state['frustration'] - 0.1)
            elif goal.progress == old_progress and goal.attempts > 3:
                # No progress is frustrating
                self.emotional_state['frustration'] = min(1.0, self.emotional_state['frustration'] + 0.1)
            
            # Check if achieved
            if goal.progress >= 1.0:
                await self._achieve_goal(goal_id)
    
    async def _achieve_goal(self, goal_id: str) -> None:
        """Отметить цель как достигнутую."""
        goal = self.goals[goal_id]
        goal.status = GoalStatus.ACHIEVED
        
        # Record achievement
        self.achievements.append((goal_id, time.time()))
        
        # Boost satisfaction and confidence
        self.emotional_state['satisfaction'] = min(1.0, self.emotional_state['satisfaction'] + 0.2)
        self.emotional_state['confidence'] = min(1.0, self.emotional_state['confidence'] + 0.1)
        self.emotional_state['frustration'] = max(0.0, self.emotional_state['frustration'] - 0.3)
        
        # Update parent goal progress
        if goal.parent_goal and goal.parent_goal in self.goals:
            parent = self.goals[goal.parent_goal]
            if parent.subgoals:
                achieved_subgoals = sum(
                    1 for sg_id in parent.subgoals
                    if sg_id in self.goals and self.goals[sg_id].status == GoalStatus.ACHIEVED
                )
                parent.progress = achieved_subgoals / len(parent.subgoals)
        
        # Remove from active
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)
        
        # Save state
        await self._save_state()
    
    async def get_next_goal(self) -> Optional[Goal]:
        """Получить следующую цель для работы (highest priority + motivation)."""
        async with self._lock:
            # Filter active or pending goals
            candidates = [
                g for g in self.goals.values()
                if g.status in (GoalStatus.ACTIVE, GoalStatus.PENDING)
            ]
            
            if not candidates:
                return None
            
            # Score by priority * motivation * (1 - progress)
            def score(goal: Goal) -> float:
                return goal.priority * goal.motivation * (1.0 - goal.progress)
            
            best_goal = max(candidates, key=score)
            
            return best_goal
    
    async def decompose_goal(self, goal_id: str) -> List[str]:
        """Декомпозировать цель на подцели."""
        async with self._lock:
            if goal_id not in self.goals:
                return []
            
            goal = self.goals[goal_id]
            
            # Simple heuristic decomposition
            desc_lower = goal.description.lower()
            
            subgoal_ids = []
            
            if 'improve' in desc_lower and 'performance' in desc_lower:
                # Decompose performance improvement
                subgoals = [
                    "Analyze current performance metrics",
                    "Identify bottlenecks",
                    "Implement optimizations",
                    "Verify improvements"
                ]
                
                for sg_desc in subgoals:
                    sg_id = await self.set_goal(
                        sg_desc,
                        priority=goal.priority - 1,
                        parent_goal=goal_id
                    )
                    subgoal_ids.append(sg_id)
            
            return subgoal_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику целей."""
        status_counts = defaultdict(int)
        for goal in self.goals.values():
            status_counts[goal.status.value] += 1
        
        return {
            'total_goals': len(self.goals),
            'active_goals': len(self.active_goals),
            'achievements': len(self.achievements),
            'status_breakdown': dict(status_counts),
            'emotional_state': self.emotional_state,
            'avg_motivation': (
                sum(g.motivation for g in self.goals.values()) / len(self.goals)
                if self.goals else 0.0
            )
        }


# Singleton
_goal_system: Optional[GoalDrivenSystem] = None
_gs_lock = asyncio.Lock()


async def get_goal_system() -> GoalDrivenSystem:
    """Get singleton goal-driven system."""
    global _goal_system
    if _goal_system is None:
        async with _gs_lock:
            if _goal_system is None:
                _goal_system = GoalDrivenSystem()
    return _goal_system


async def set_goal(description: str, priority: int = 5) -> str:
    """Set a new goal."""
    gs = await get_goal_system()
    return await gs.set_goal(description, priority)


async def get_current_goal() -> Optional[Goal]:
    """Get current goal to work on."""
    gs = await get_goal_system()
    return await gs.get_next_goal()


async def mark_progress(goal_id: str, progress: float) -> None:
    """Update goal progress."""
    gs = await get_goal_system()
    await gs.update_progress(goal_id, progress)


__all__ = [
    "GoalDrivenSystem",
    "Goal",
    "GoalStatus",
    "get_goal_system",
    "set_goal",
    "get_current_goal",
    "mark_progress",
]
