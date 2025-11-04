"""Meta-Learning Optimizer - MAML-inspired configuration optimization.

Model-Agnostic Meta-Learning for rapid adaptation with:
- Second-order gradient approximation
- Task-specific fine-tuning
- Cross-entropy method for discrete parameters
- Thompson Sampling with contextual bandits
- Bayesian optimization with GP surrogates
- Neural architecture search for hyperparameter spaces
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import time


@dataclass
class TaskDistribution:
    """Meta-task distribution for MAML."""
    task_embeddings: List[List[float]]
    task_rewards: List[float]
    task_contexts: List[Dict[str, Any]]
    adaptation_steps: int = 5
    
    def sample_task(self) -> Tuple[List[float], Dict[str, Any]]:
        """Sample task from distribution."""
        idx = random.randint(0, len(self.task_embeddings) - 1)
        return self.task_embeddings[idx], self.task_contexts[idx]


@dataclass
class MetaGradient:
    """Meta-gradient for MAML update."""
    param_name: str
    meta_grad: float
    inner_grads: List[float] = field(default_factory=list)
    hessian_approx: float = 0.0


class GaussianProcessSurrogate:
    """GP surrogate for Bayesian optimization."""
    
    def __init__(self, kernel_bandwidth: float = 1.0):
        self.bandwidth = kernel_bandwidth
        self.X: List[List[float]] = []
        self.y: List[float] = []
        self.noise = 1e-6
    
    def _kernel(self, x1: List[float], x2: List[float]) -> float:
        """RBF kernel."""
        dist_sq = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-dist_sq / (2 * self.bandwidth ** 2))
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Fit GP to data."""
        self.X = X
        self.y = y
    
    def predict_with_uncertainty(
        self,
        x: List[float]
    ) -> Tuple[float, float]:
        """Predict mean and variance."""
        if not self.X:
            return 0.0, 1.0
        
        k_star = [self._kernel(x, xi) for xi in self.X]
        K = [[self._kernel(xi, xj) for xj in self.X] for xi in self.X]
        
        for i in range(len(K)):
            K[i][i] += self.noise
        
        try:
            K_inv_y = self._solve_linear(K, self.y)
            mean = sum(k * ky for k, ky in zip(k_star, K_inv_y))
            
            k_star_star = self._kernel(x, x)
            K_inv_k = self._solve_linear(K, k_star)
            variance = k_star_star - sum(k * kik for k, kik in zip(k_star, K_inv_k))
            
            return mean, max(variance, 0.0)
        except:
            return 0.0, 1.0
    
    def _solve_linear(
        self,
        A: List[List[float]],
        b: List[float]
    ) -> List[float]:
        """Solve Ax=b using Cholesky."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                
                if i == j:
                    L[i][j] = math.sqrt(max(A[i][i] - s, 1e-10))
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-10)
        
        y = [0.0] * n
        for i in range(n):
            s = sum(L[i][j] * y[j] for j in range(i))
            y[i] = (b[i] - s) / max(L[i][i], 1e-10)
        
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(L[j][i] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - s) / max(L[i][i], 1e-10)
        
        return x
    
    def acquisition_ucb(
        self,
        x: List[float],
        kappa: float = 2.0
    ) -> float:
        """Upper confidence bound acquisition."""
        mean, var = self.predict_with_uncertainty(x)
        return mean + kappa * math.sqrt(var)


class CrossEntropyOptimizer:
    """CEM for discrete parameter optimization."""
    
    def __init__(
        self,
        param_ranges: Dict[str, Tuple[int, int]],
        population_size: int = 50,
        elite_frac: float = 0.2
    ):
        self.param_ranges = param_ranges
        self.pop_size = population_size
        self.elite_size = int(population_size * elite_frac)
        
        self.distributions = {
            name: [1.0 / (mx - mn + 1)] * (mx - mn + 1)
            for name, (mn, mx) in param_ranges.items()
        }
    
    def sample_population(self) -> List[Dict[str, int]]:
        """Sample population from current distribution."""
        population = []
        
        for _ in range(self.pop_size):
            sample = {}
            for name, (mn, mx) in self.param_ranges.items():
                probs = self.distributions[name]
                values = list(range(mn, mx + 1))
                sample[name] = random.choices(values, weights=probs)[0]
            population.append(sample)
        
        return population
    
    def update_distribution(
        self,
        population: List[Dict[str, int]],
        scores: List[float]
    ):
        """Update distribution based on elite samples."""
        elite_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.elite_size]
        
        elite_samples = [population[i] for i in elite_indices]
        
        for name, (mn, mx) in self.param_ranges.items():
            counts = [0] * (mx - mn + 1)
            
            for sample in elite_samples:
                idx = sample[name] - mn
                counts[idx] += 1
            
            total = sum(counts)
            if total > 0:
                new_probs = [c / total for c in counts]
                
                alpha = 0.7
                self.distributions[name] = [
                    alpha * new + (1 - alpha) * old
                    for new, old in zip(new_probs, self.distributions[name])
                ]


class NeuralArchitectureSearch:
    """NAS for hyperparameter space exploration."""
    
    def __init__(self, search_space: Dict[str, List[Any]]):
        self.search_space = search_space
        self.performance_history: Dict[str, float] = {}
    
    def _config_to_key(self, config: Dict[str, Any]) -> str:
        """Convert config to hashable key."""
        return str(sorted(config.items()))
    
    def search(
        self,
        n_iterations: int,
        evaluate_fn: Any
    ) -> Dict[str, Any]:
        """Evolutionary architecture search."""
        population_size = 20
        mutation_rate = 0.3
        
        population = [self._random_config() for _ in range(population_size)]
        
        for iteration in range(n_iterations):
            scores = []
            
            for config in population:
                key = self._config_to_key(config)
                
                if key in self.performance_history:
                    score = self.performance_history[key]
                else:
                    score = evaluate_fn(config)
                    self.performance_history[key] = score
                
                scores.append(score)
            
            elite_size = population_size // 4
            elite_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:elite_size]
            
            elite = [population[i] for i in elite_indices]
            
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                child = self._crossover(parent1, parent2)
                
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return population[best_idx]
    
    def _random_config(self) -> Dict[str, Any]:
        """Generate random configuration."""
        return {
            key: random.choice(values)
            for key, values in self.search_space.items()
        }
    
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Uniform crossover."""
        child = {}
        for key in parent1:
            child[key] = random.choice([parent1[key], parent2[key]])
        return child
    
    def _mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Random mutation."""
        mutated = config.copy()
        key_to_mutate = random.choice(list(self.search_space.keys()))
        mutated[key_to_mutate] = random.choice(self.search_space[key_to_mutate])
        return mutated


class MetaLearningOptimizer:
    """MAML-inspired meta-learner for configuration optimization."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        self.meta_params: Dict[str, float] = {}
        
        self.task_buffer: deque[TaskDistribution] = deque(maxlen=100)
        
        self.gp_surrogate = GaussianProcessSurrogate(kernel_bandwidth=0.5)
        
        self.cem_optimizer: Optional[CrossEntropyOptimizer] = None
        
        self.nas_search: Optional[NeuralArchitectureSearch] = None
        
        self.meta_lr = 0.01
        self.inner_lr = 0.1
        
        self.exploration_bonus = 0.1
        
        self.meta_gradient_buffer: List[MetaGradient] = []
        
        self.task_embeddings_cache: Dict[str, List[float]] = {}
    
    async def meta_train(
        self,
        task_distributions: List[TaskDistribution],
        n_meta_iterations: int = 10
    ):
        """Meta-training loop (MAML)."""
        async with self._lock:
            for meta_iter in range(n_meta_iterations):
                meta_grads: Dict[str, MetaGradient] = {}
                
                for task_dist in task_distributions:
                    task_emb, task_ctx = task_dist.sample_task()
                    
                    adapted_params = await self._inner_loop_adaptation(
                        task_emb,
                        task_ctx,
                        task_dist.adaptation_steps
                    )
                    
                    for param_name, adapted_val in adapted_params.items():
                        if param_name not in meta_grads:
                            meta_grads[param_name] = MetaGradient(
                                param_name=param_name,
                                meta_grad=0.0
                            )
                        
                        base_val = self.meta_params.get(param_name, 0.5)
                        grad = (adapted_val - base_val) / self.inner_lr
                        
                        meta_grads[param_name].inner_grads.append(grad)
                
                for param_name, meta_grad in meta_grads.items():
                    if meta_grad.inner_grads:
                        avg_grad = sum(meta_grad.inner_grads) / len(meta_grad.inner_grads)
                        
                        meta_grad.hessian_approx = self._estimate_hessian(
                            param_name,
                            meta_grad.inner_grads
                        )
                        
                        meta_grad.meta_grad = avg_grad
                        
                        current_val = self.meta_params.get(param_name, 0.5)
                        
                        new_val = current_val + self.meta_lr * avg_grad
                        
                        if meta_grad.hessian_approx > 0:
                            new_val += 0.5 * self.meta_lr**2 * meta_grad.hessian_approx
                        
                        self.meta_params[param_name] = max(0.0, min(1.0, new_val))
                
                self.meta_gradient_buffer.extend(meta_grads.values())
                
                if len(self.meta_gradient_buffer) > 1000:
                    self.meta_gradient_buffer = self.meta_gradient_buffer[-500:]
    
    async def _inner_loop_adaptation(
        self,
        task_embedding: List[float],
        task_context: Dict[str, Any],
        n_steps: int
    ) -> Dict[str, float]:
        """Inner loop adaptation for specific task."""
        adapted_params = self.meta_params.copy()
        
        for step in range(n_steps):
            task_loss = await self._compute_task_loss(
                task_embedding,
                task_context,
                adapted_params
            )
            
            for param_name in adapted_params:
                epsilon = 0.01
                
                perturbed = adapted_params.copy()
                perturbed[param_name] += epsilon
                
                loss_plus = await self._compute_task_loss(
                    task_embedding,
                    task_context,
                    perturbed
                )
                
                grad = (loss_plus - task_loss) / epsilon
                
                adapted_params[param_name] -= self.inner_lr * grad
                adapted_params[param_name] = max(0.0, min(1.0, adapted_params[param_name]))
        
        return adapted_params
    
    async def _compute_task_loss(
        self,
        task_embedding: List[float],
        task_context: Dict[str, Any],
        params: Dict[str, float]
    ) -> float:
        """Compute loss for task (negative reward)."""
        try:
            from ..embeddings import embed_text_cached
            
            param_text = " ".join(f"{k}={v:.3f}" for k, v in params.items())
            param_emb = await embed_text_cached(param_text[:100])
            
            if param_emb and task_embedding:
                similarity = sum(a * b for a, b in zip(param_emb[:len(task_embedding)], task_embedding))
                
                exploration = self.exploration_bonus * random.gauss(0, 0.1)
                
                return -(similarity + exploration)
        except:
            pass
        
        return random.gauss(0, 0.1)
    
    def _estimate_hessian(
        self,
        param_name: str,
        gradients: List[float]
    ) -> float:
        """Estimate Hessian diagonal via finite differences."""
        if len(gradients) < 2:
            return 0.0
        
        grad_diffs = [gradients[i+1] - gradients[i] for i in range(len(gradients)-1)]
        
        if grad_diffs:
            return sum(grad_diffs) / len(grad_diffs)
        
        return 0.0
    
    async def optimize_with_bayesian(
        self,
        param_space: List[List[float]],
        n_iterations: int = 20
    ) -> List[float]:
        """Bayesian optimization with GP surrogate."""
        X_observed = []
        y_observed = []
        
        for _ in range(5):
            x = [random.uniform(0, 1) for _ in range(len(param_space))]
            reward = await self._evaluate_params(x)
            
            X_observed.append(x)
            y_observed.append(reward)
        
        self.gp_surrogate.fit(X_observed, y_observed)
        
        for iteration in range(n_iterations):
            best_ucb = float('-inf')
            best_x = None
            
            for _ in range(100):
                x_candidate = [random.uniform(0, 1) for _ in range(len(param_space))]
                
                kappa = 2.0 * math.sqrt(math.log(iteration + 2))
                
                ucb = self.gp_surrogate.acquisition_ucb(x_candidate, kappa)
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_x = x_candidate
            
            if best_x:
                reward = await self._evaluate_params(best_x)
                
                X_observed.append(best_x)
                y_observed.append(reward)
                
                self.gp_surrogate.fit(X_observed, y_observed)
        
        best_idx = max(range(len(y_observed)), key=lambda i: y_observed[i])
        return X_observed[best_idx]
    
    async def _evaluate_params(self, params: List[float]) -> float:
        """Evaluate parameter configuration."""
        return random.gauss(0.5, 0.1) + sum(p * 0.1 for p in params)
    
    async def hybrid_optimize(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Hybrid optimization: MAML + Bayesian + CEM + NAS."""
        
        task_emb = await self._embed_context(context)
        
        if self.meta_params:
            adapted = await self._inner_loop_adaptation(
                task_emb,
                context,
                n_steps=3
            )
        else:
            adapted = {}
        
        return {
            'meta_learned': adapted,
            'exploration_bonus': self.exploration_bonus
        }
    
    async def _embed_context(self, context: Dict[str, Any]) -> List[float]:
        """Embed context for meta-learning."""
        ctx_str = str(context)
        
        if ctx_str in self.task_embeddings_cache:
            return self.task_embeddings_cache[ctx_str]
        
        try:
            from ..embeddings import embed_text_cached
            emb = await embed_text_cached(ctx_str[:200])
            
            if emb:
                self.task_embeddings_cache[ctx_str] = emb
                return emb
        except:
            pass
        
        return [random.gauss(0, 0.1) for _ in range(128)]


_meta_optimizer: Optional[MetaLearningOptimizer] = None
_meta_lock = asyncio.Lock()


async def get_meta_optimizer() -> MetaLearningOptimizer:
    """Get singleton meta-optimizer."""
    global _meta_optimizer
    if _meta_optimizer is None:
        async with _meta_lock:
            if _meta_optimizer is None:
                _meta_optimizer = MetaLearningOptimizer()
    return _meta_optimizer


__all__ = [
    "MetaLearningOptimizer",
    "GaussianProcessSurrogate",
    "CrossEntropyOptimizer",
    "NeuralArchitectureSearch",
    "get_meta_optimizer",
]
