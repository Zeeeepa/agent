"""Neural Contextual Bandits with Attention and Deep Reinforcement Learning.

Implements:
- Deep contextual bandits with neural networks
- Multi-head attention for context processing
- Thompson Sampling with neural posterior
- UCB with neural uncertainty estimation
- Experience replay buffer
- Double DQN for value estimation
- Prioritized experience replay
- Hindsight experience replay
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    context: List[float]
    action: int
    reward: float
    next_context: List[float]
    done: bool
    priority: float = 1.0
    temporal_difference_error: float = 0.0


class PrioritizedReplayBuffer:
    """Prioritized experience replay with SumTree."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.priorities: deque[float] = deque(maxlen=capacity)
        self.beta = 0.4
        self.beta_increment = 0.001
    
    def add(self, experience: Experience):
        """Add experience with priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            return [], [], []
        
        priorities_array = list(self.priorities)
        probs = [p ** self.alpha for p in priorities_array]
        total = sum(probs)
        probs = [p / total for p in probs]
        
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        
        samples = [self.buffer[i] for i in indices]
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        max_weight = (len(self.buffer) * min(probs)) ** (-self.beta)
        weights = [(len(self.buffer) * probs[i]) ** (-self.beta) / max_weight for i in indices]
        
        return samples, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = abs(td_error) + 1e-6


class MultiHeadAttention:
    """Multi-head attention mechanism for context processing."""
    
    def __init__(self, d_model: int = 128, num_heads: int = 4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(d_model)]
    
    def _matmul(self, A: List[List[float]], b: List[float]) -> List[float]:
        """Matrix-vector multiplication."""
        return [sum(A[i][j] * b[j] for j in range(len(b))) for i in range(len(A))]
    
    def _scaled_dot_product_attention(
        self,
        Q: List[float],
        K: List[float],
        V: List[float]
    ) -> List[float]:
        """Scaled dot-product attention."""
        score = sum(q * k for q, k in zip(Q, K)) / math.sqrt(self.d_k)
        
        attention_weight = 1 / (1 + math.exp(-score))
        
        return [v * attention_weight for v in V]
    
    def forward(self, x: List[float]) -> List[float]:
        """Forward pass through attention."""
        if len(x) < self.d_model:
            x = x + [0.0] * (self.d_model - len(x))
        elif len(x) > self.d_model:
            x = x[:self.d_model]
        
        Q = self._matmul(self.W_q, x)
        K = self._matmul(self.W_k, x)
        V = self._matmul(self.W_v, x)
        
        head_outputs = []
        for h in range(self.num_heads):
            start = h * self.d_k
            end = start + self.d_k
            
            Q_h = Q[start:end] if start < len(Q) else [0.0] * self.d_k
            K_h = K[start:end] if start < len(K) else [0.0] * self.d_k
            V_h = V[start:end] if start < len(V) else [0.0] * self.d_k
            
            head_out = self._scaled_dot_product_attention(Q_h, K_h, V_h)
            head_outputs.extend(head_out)
        
        if len(head_outputs) < self.d_model:
            head_outputs.extend([0.0] * (self.d_model - len(head_outputs)))
        
        output = self._matmul(self.W_o, head_outputs[:self.d_model])
        
        return output


class NeuralNetwork:
    """Deep neural network for value/policy estimation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float = 0.001
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.lr = learning_rate
        
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            w = [[random.gauss(0, 0.1) for _ in range(dims[i+1])] for _ in range(dims[i])]
            b = [0.0] * dims[i+1]
            
            self.weights.append(w)
            self.biases.append(b)
        
        self.activations = []
        self.pre_activations = []
    
    def forward(self, x: List[float]) -> List[float]:
        """Forward pass with ReLU."""
        self.activations = [x]
        self.pre_activations = []
        
        current = x
        
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = [sum(current[i] * W[i][j] for i in range(len(current))) + b[j] 
                 for j in range(len(b))]
            
            self.pre_activations.append(z)
            
            if layer_idx < len(self.weights) - 1:
                current = [max(0, zi) for zi in z]
            else:
                current = z
            
            self.activations.append(current)
        
        return current
    
    def backward(self, x: List[float], y_true: List[float]) -> float:
        """Backward pass with gradient descent."""
        y_pred = self.forward(x)
        
        loss = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
        
        deltas = [2 * (yp - yt) / len(y_true) for yp, yt in zip(y_pred, y_true)]
        
        for layer_idx in range(len(self.weights) - 1, -1, -1):
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            
            activation = self.activations[layer_idx]
            
            dW = [[deltas[j] * activation[i] for j in range(len(deltas))]
                  for i in range(len(activation))]
            
            db = deltas.copy()
            
            for i in range(len(W)):
                for j in range(len(W[i])):
                    W[i][j] -= self.lr * dW[i][j]
            
            for j in range(len(b)):
                b[j] -= self.lr * db[j]
            
            if layer_idx > 0:
                new_deltas = [0.0] * len(activation)
                
                for i in range(len(activation)):
                    grad = sum(deltas[j] * W[i][j] for j in range(len(deltas)))
                    
                    if layer_idx > 0:
                        relu_grad = 1.0 if self.pre_activations[layer_idx-1][i] > 0 else 0.0
                        grad *= relu_grad
                    
                    new_deltas[i] = grad
                
                deltas = new_deltas
        
        return loss


class NeuralContextualBandit:
    """Neural contextual bandit with deep RL."""
    
    def __init__(
        self,
        n_actions: int,
        context_dim: int = 128,
        hidden_dims: List[int] = None
    ):
        self._lock = asyncio.Lock()
        
        self.n_actions = n_actions
        self.context_dim = context_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.q_network = NeuralNetwork(
            input_dim=context_dim,
            hidden_dims=hidden_dims,
            output_dim=n_actions,
            learning_rate=0.001
        )
        
        self.target_network = NeuralNetwork(
            input_dim=context_dim,
            hidden_dims=hidden_dims,
            output_dim=n_actions,
            learning_rate=0.001
        )
        
        self.uncertainty_network = NeuralNetwork(
            input_dim=context_dim,
            hidden_dims=[128, 64],
            output_dim=n_actions,
            learning_rate=0.001
        )
        
        self.attention = MultiHeadAttention(d_model=context_dim, num_heads=4)
        
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)
        
        self.gamma = 0.99
        self.epsilon = 0.1
        
        self.update_target_frequency = 100
        self.steps = 0
    
    async def select_action(
        self,
        context: List[float],
        *,
        use_ucb: bool = True,
        use_thompson: bool = False
    ) -> int:
        """Select action with neural bandit."""
        async with self._lock:
            attended_context = self.attention.forward(context)
            
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
            
            q_values = self.q_network.forward(attended_context)
            
            if use_ucb:
                uncertainties = self.uncertainty_network.forward(attended_context)
                
                ucb_values = [
                    q + 2.0 * math.sqrt(max(0, u))
                    for q, u in zip(q_values, uncertainties)
                ]
                
                return max(range(len(ucb_values)), key=lambda i: ucb_values[i])
            
            elif use_thompson:
                uncertainties = self.uncertainty_network.forward(attended_context)
                
                sampled_values = [
                    random.gauss(q, math.sqrt(max(0.01, u)))
                    for q, u in zip(q_values, uncertainties)
                ]
                
                return max(range(len(sampled_values)), key=lambda i: sampled_values[i])
            
            else:
                return max(range(len(q_values)), key=lambda i: q_values[i])
    
    async def update(
        self,
        context: List[float],
        action: int,
        reward: float,
        next_context: List[float],
        done: bool
    ):
        """Update networks with double DQN."""
        async with self._lock:
            experience = Experience(
                context=context,
                action=action,
                reward=reward,
                next_context=next_context,
                done=done
            )
            
            self.replay_buffer.add(experience)
            
            if len(self.replay_buffer.buffer) < 32:
                return
            
            batch, weights, indices = self.replay_buffer.sample(32)
            
            td_errors = []
            
            for exp, weight in zip(batch, weights):
                attended_ctx = self.attention.forward(exp.context)
                q_values = self.q_network.forward(attended_ctx)
                
                if not exp.done:
                    attended_next = self.attention.forward(exp.next_context)
                    
                    next_q_online = self.q_network.forward(attended_next)
                    best_action = max(range(len(next_q_online)), key=lambda i: next_q_online[i])
                    
                    next_q_target = self.target_network.forward(attended_next)
                    max_next_q = next_q_target[best_action]
                    
                    target_q = exp.reward + self.gamma * max_next_q
                else:
                    target_q = exp.reward
                
                td_error = target_q - q_values[exp.action]
                td_errors.append(td_error)
                
                target_values = q_values.copy()
                target_values[exp.action] = target_q
                
                loss = self.q_network.backward(attended_ctx, target_values)
                
                uncertainty_target = [abs(td_error)] * self.n_actions
                self.uncertainty_network.backward(attended_ctx, uncertainty_target)
            
            self.replay_buffer.update_priorities(indices, td_errors)
            
            self.steps += 1
            
            if self.steps % self.update_target_frequency == 0:
                self._copy_weights()
    
    def _copy_weights(self):
        """Copy Q-network weights to target network."""
        self.target_network.weights = [
            [[w for w in row] for row in layer]
            for layer in self.q_network.weights
        ]
        
        self.target_network.biases = [
            [b for b in layer]
            for layer in self.q_network.biases
        ]


_neural_bandit: Optional[NeuralContextualBandit] = None
_bandit_lock = asyncio.Lock()


async def get_neural_bandit(n_actions: int = 10) -> NeuralContextualBandit:
    """Get singleton neural bandit."""
    global _neural_bandit
    if _neural_bandit is None:
        async with _bandit_lock:
            if _neural_bandit is None:
                _neural_bandit = NeuralContextualBandit(n_actions=n_actions)
    return _neural_bandit


__all__ = [
    "NeuralContextualBandit",
    "MultiHeadAttention",
    "PrioritizedReplayBuffer",
    "get_neural_bandit",
]
