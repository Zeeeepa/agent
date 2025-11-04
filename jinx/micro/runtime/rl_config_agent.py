"""Reinforcement Learning Config Agent - Long-term optimization using Q-learning/Policy Gradient.

Uses:
- Q-Learning with experience replay
- Policy gradient for continuous actions
- Multi-step returns (n-step TD)
- Prioritized experience replay
- Target networks for stability
"""

from __future__ import annotations

import asyncio
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray  # Current config + context
    action: np.ndarray  # Config adjustment
    reward: float  # Quality improvement
    next_state: np.ndarray
    done: bool
    priority: float = 1.0


class QNetwork:
    """
    Simplified Q-Network for config selection.
    
    Maps (state, action) -> expected quality
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple linear approximation (can be replaced with NN)
        self.W = np.random.randn(state_dim + action_dim, 1) * 0.01
        self.b = 0.0
        
        # Learning rate
        self.lr = 0.01
    
    def predict(self, state: np.ndarray, action: np.ndarray) -> float:
        """Predict Q-value for (state, action) pair."""
        x = np.concatenate([state, action])
        return float(x.dot(self.W) + self.b)
    
    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        target: float
    ):
        """Update weights using gradient descent."""
        x = np.concatenate([state, action])
        
        # Forward pass
        pred = x.dot(self.W) + self.b
        
        # Error
        error = pred - target
        
        # Gradient descent
        self.W -= self.lr * error * x.reshape(-1, 1)
        self.b -= self.lr * error


class RLConfigAgent:
    """
    Reinforcement Learning agent for long-term config optimization.
    
    Features:
    - Q-Learning with experience replay
    - Multi-step returns (n-step TD)
    - Prioritized experience replay
    - Target network for stability
    - Epsilon-greedy exploration
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # State: [normalized_config (5D), task_one_hot (9D), recent_performance (3D)]
        self.state_dim = 5 + 9 + 3  # 17D
        
        # Action: config adjustments [-1, +1] for each parameter
        self.action_dim = 5  # 5 tunable parameters
        
        # Q-Networks
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.target_network = QNetwork(self.state_dim, self.action_dim)
        
        # Copy weights to target
        self.target_network.W = self.q_network.W.copy()
        self.target_network.b = self.q_network.b
        
        # Experience replay buffer
        self.replay_buffer: deque[Experience] = deque(maxlen=10000)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.batch_size = 32
        self.target_update_frequency = 100
        self.update_counter = 0
        
        # N-step returns
        self.n_step = 3
        self.n_step_buffer: deque = deque(maxlen=self.n_step)
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        
        # Current state
        self.current_state: Optional[np.ndarray] = None
        self.current_action: Optional[np.ndarray] = None
        
        # Task types mapping
        self.task_types = [
            'code_search', 'code_analysis', 'debugging', 'refactoring',
            'implementation', 'testing', 'documentation', 'planning', 'conversation'
        ]
    
    def _encode_state(
        self,
        config: Dict[str, float],
        task_type: str,
        recent_performance: Dict[str, float]
    ) -> np.ndarray:
        """Encode state as vector."""
        
        # Normalize config (5D)
        config_vec = np.array([
            config.get('EMBED_PROJECT_TOP_K', 50) / 150,
            config.get('EMBED_PROJECT_SCORE_THRESHOLD', 0.15) / 0.40,
            config.get('JINX_STAGE_PROJCTX_MS', 5000) / 15000,
            config.get('JINX_MAX_CONCURRENT', 6) / 16,
            float(config.get('EMBED_PROJECT_EXHAUSTIVE', 1))
        ], dtype=np.float32)
        
        # One-hot encode task type (9D)
        task_idx = self.task_types.index(task_type) if task_type in self.task_types else 0
        task_vec = np.zeros(9, dtype=np.float32)
        task_vec[task_idx] = 1.0
        
        # Recent performance (3D): success_rate, avg_latency_norm, quality
        perf_vec = np.array([
            recent_performance.get('success_rate', 0.5),
            1.0 - min(1.0, recent_performance.get('avg_latency', 3000) / 10000),
            recent_performance.get('quality', 0.5)
        ], dtype=np.float32)
        
        # Concatenate
        state = np.concatenate([config_vec, task_vec, perf_vec])
        
        return state
    
    def _sample_action(self, state: np.ndarray, greedy: bool = False) -> np.ndarray:
        """
        Sample action using epsilon-greedy policy.
        
        Returns:
            action: [-1, +1] adjustment for each parameter
        """
        
        if not greedy and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
        
        # Exploit: sample multiple actions and choose best Q-value
        num_samples = 10
        best_action = None
        best_q = float('-inf')
        
        for _ in range(num_samples):
            action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            q = self.q_network.predict(state, action)
            
            if q > best_q:
                best_q = q
                best_action = action
        
        return best_action
    
    async def select_action(
        self,
        config: Dict[str, float],
        task_type: str,
        recent_performance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Select config adjustment action.
        
        Returns:
            Adjusted configuration
        """
        
        async with self._lock:
            # Encode state
            state = self._encode_state(config, task_type, recent_performance)
            
            # Sample action
            action = self._sample_action(state)
            
            # Store for later
            self.current_state = state
            self.current_action = action
            
            # Apply action to config
            adjusted_config = self._apply_action(config, action)
            
            return adjusted_config
    
    def _apply_action(
        self,
        config: Dict[str, float],
        action: np.ndarray
    ) -> Dict[str, float]:
        """Apply action (adjustments) to config."""
        
        new_config = config.copy()
        
        # Apply adjustments with constraints
        params = [
            'EMBED_PROJECT_TOP_K',
            'EMBED_PROJECT_SCORE_THRESHOLD',
            'JINX_STAGE_PROJCTX_MS',
            'JINX_MAX_CONCURRENT',
            'EMBED_PROJECT_EXHAUSTIVE'
        ]
        
        constraints = {
            'EMBED_PROJECT_TOP_K': (10, 150),
            'EMBED_PROJECT_SCORE_THRESHOLD': (0.05, 0.40),
            'JINX_STAGE_PROJCTX_MS': (1000, 15000),
            'JINX_MAX_CONCURRENT': (1, 16),
            'EMBED_PROJECT_EXHAUSTIVE': (0, 1)
        }
        
        for i, param in enumerate(params):
            current = new_config.get(param, 50)
            min_val, max_val = constraints[param]
            
            # Adjustment proportional to range
            adjustment = action[i] * (max_val - min_val) * 0.1  # 10% max change
            
            new_value = current + adjustment
            new_value = max(min_val, min(max_val, new_value))
            
            new_config[param] = new_value
        
        return new_config
    
    async def observe_reward(
        self,
        reward: float,
        next_config: Dict[str, float],
        next_task_type: str,
        next_performance: Dict[str, float],
        done: bool = False
    ):
        """
        Observe reward and next state.
        
        Args:
            reward: Quality score or improvement
            next_config: Configuration after action
            next_task_type: Next task type
            next_performance: Performance metrics
            done: Episode ended
        """
        
        async with self._lock:
            if self.current_state is None or self.current_action is None:
                return
            
            # Encode next state
            next_state = self._encode_state(next_config, next_task_type, next_performance)
            
            # Add to n-step buffer
            self.n_step_buffer.append({
                'state': self.current_state,
                'action': self.current_action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            # If buffer full, create experience with n-step return
            if len(self.n_step_buffer) >= self.n_step or done:
                experience = self._create_n_step_experience()
                if experience:
                    self.replay_buffer.append(experience)
            
            # Track episode reward
            self.current_episode_reward += reward
            
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0.0
                self.n_step_buffer.clear()
            
            # Update current state
            self.current_state = next_state
            
            # Train if enough experiences
            if len(self.replay_buffer) >= self.batch_size:
                await self._train_step()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _create_n_step_experience(self) -> Optional[Experience]:
        """Create experience with n-step return."""
        
        if not self.n_step_buffer:
            return None
        
        # First transition
        first = self.n_step_buffer[0]
        
        # Compute n-step return
        n_step_return = 0.0
        for i, transition in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * transition['reward']
        
        # Last transition
        last = self.n_step_buffer[-1]
        
        # Priority based on reward magnitude
        priority = abs(n_step_return) + 1e-6
        
        return Experience(
            state=first['state'],
            action=first['action'],
            reward=n_step_return,
            next_state=last['next_state'],
            done=last['done'],
            priority=priority
        )
    
    async def _train_step(self):
        """Perform one training step using experience replay."""
        
        # Prioritized sampling
        experiences = list(self.replay_buffer)
        priorities = np.array([exp.priority for exp in experiences])
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(
            len(experiences),
            size=min(self.batch_size, len(experiences)),
            replace=False,
            p=probs
        )
        
        batch = [experiences[i] for i in indices]
        
        # Update Q-network for each experience
        for exp in batch:
            # Compute target
            if exp.done:
                target = exp.reward
            else:
                # Sample actions for next state
                next_action = self._sample_action(exp.next_state, greedy=True)
                next_q = self.target_network.predict(exp.next_state, next_action)
                target = exp.reward + self.gamma ** self.n_step * next_q
            
            # Update Q-network
            self.q_network.update(exp.state, exp.action, target)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.W = self.q_network.W.copy()
            self.target_network.b = self.q_network.b
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RL agent statistics."""
        
        return {
            'replay_buffer_size': len(self.replay_buffer),
            'epsilon': self.epsilon,
            'episodes': len(self.episode_rewards),
            'avg_episode_reward': (
                float(np.mean(self.episode_rewards[-50:]))
                if self.episode_rewards else 0.0
            ),
            'update_counter': self.update_counter,
            'n_step': self.n_step
        }


# Singleton
_rl_agent: Optional[RLConfigAgent] = None
_rl_lock = asyncio.Lock()


async def get_rl_agent() -> RLConfigAgent:
    """Get singleton RL agent."""
    global _rl_agent
    if _rl_agent is None:
        async with _rl_lock:
            if _rl_agent is None:
                _rl_agent = RLConfigAgent()
    return _rl_agent


__all__ = [
    "RLConfigAgent",
    "get_rl_agent",
]
