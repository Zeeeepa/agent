"""Bayesian Configuration Optimizer - Advanced parameter optimization using Gaussian Processes.

Uses:
- Gaussian Process regression for performance modeling
- Expected Improvement acquisition function
- Thompson Sampling for exploration
- Multi-objective optimization (latency + success_rate)
- Contextual bandits for per-task optimization
"""

from __future__ import annotations

import asyncio
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class ConfigObservation:
    """Single observation of config performance."""
    config: Dict[str, float]  # Normalized [0, 1]
    context: str  # Task type
    latency_ms: float
    success_rate: float
    quality_score: float
    timestamp: float


class GaussianProcess:
    """
    Simplified Gaussian Process for performance prediction.
    
    Predicts quality_score given configuration vector.
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        
        # Kernel hyperparameters
        self.length_scale = 0.3
        self.noise_std = 0.1
        
        # Training data
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []
        
        # Cached kernel matrix inverse
        self._K_inv: Optional[np.ndarray] = None
        self._need_update = True
    
    def add_observation(self, x: np.ndarray, y: float):
        """Add training point."""
        self.X_train.append(x)
        self.y_train.append(y)
        self._need_update = True
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBF (squared exponential) kernel."""
        diff = x1 - x2
        sq_dist = np.sum(diff ** 2)
        return np.exp(-sq_dist / (2 * self.length_scale ** 2))
    
    def _compute_kernel_matrix(self) -> np.ndarray:
        """Compute kernel matrix K(X, X)."""
        n = len(self.X_train)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                k = self._rbf_kernel(self.X_train[i], self.X_train[j])
                K[i, j] = k
                K[j, i] = k
        
        # Add noise to diagonal
        K += (self.noise_std ** 2) * np.eye(n)
        
        return K
    
    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Predict mean and std at point x.
        
        Returns:
            (mean, std)
        """
        
        if not self.X_train:
            return (0.5, 1.0)  # Prior
        
        # Update kernel matrix if needed
        if self._need_update:
            K = self._compute_kernel_matrix()
            try:
                self._K_inv = np.linalg.inv(K)
                self._need_update = False
            except np.linalg.LinAlgError:
                # Singular matrix - add more noise
                K += 1e-6 * np.eye(len(K))
                self._K_inv = np.linalg.inv(K)
                self._need_update = False
        
        # Compute k(x, X)
        k = np.array([self._rbf_kernel(x, x_train) for x_train in self.X_train])
        
        # Compute k(x, x)
        k_xx = self._rbf_kernel(x, x)
        
        # Mean prediction
        y_arr = np.array(self.y_train)
        mean = k.dot(self._K_inv).dot(y_arr)
        
        # Variance prediction
        variance = k_xx - k.dot(self._K_inv).dot(k)
        variance = max(1e-8, variance)  # Ensure positive
        std = np.sqrt(variance)
        
        return (float(mean), float(std))


class BayesianConfigOptimizer:
    """
    Bayesian optimization for configuration parameters.
    
    Features:
    - Gaussian Process regression
    - Expected Improvement acquisition
    - Thompson Sampling
    - Contextual optimization (per task type)
    - Multi-objective (latency + success_rate)
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Parameter space definition
        self._param_space = {
            'EMBED_PROJECT_TOP_K': {
                'min': 10,
                'max': 150,
                'type': 'int',
                'default': 50
            },
            'EMBED_PROJECT_SCORE_THRESHOLD': {
                'min': 0.05,
                'max': 0.40,
                'type': 'float',
                'default': 0.15
            },
            'JINX_STAGE_PROJCTX_MS': {
                'min': 1000,
                'max': 15000,
                'type': 'int',
                'default': 5000
            },
            'JINX_MAX_CONCURRENT': {
                'min': 1,
                'max': 16,
                'type': 'int',
                'default': 6
            },
            'EMBED_PROJECT_EXHAUSTIVE': {
                'min': 0,
                'max': 1,
                'type': 'bool',
                'default': 1
            }
        }
        
        self._param_names = list(self._param_space.keys())
        self._dim = len(self._param_names)
        
        # GP models per context (task type)
        self._gp_models: Dict[str, GaussianProcess] = {}
        
        # Observation history
        self._observations: deque[ConfigObservation] = deque(maxlen=1000)
        
        # Best configs found per context
        self._best_configs: Dict[str, Tuple[Dict[str, float], float]] = {}
        
        # Exploration vs exploitation parameter
        self._exploration_rate = 0.2  # 20% exploration
        
        # Thompson sampling parameters
        self._thompson_samples = 10
    
    def _normalize_config(self, config: Dict[str, Any]) -> np.ndarray:
        """Normalize config to [0, 1] vector."""
        
        vec = np.zeros(self._dim, dtype=np.float32)
        
        for i, param in enumerate(self._param_names):
            value = config.get(param)
            spec = self._param_space[param]
            
            if value is None:
                value = spec['default']
            
            # Normalize to [0, 1]
            if spec['type'] == 'bool':
                vec[i] = float(value)
            else:
                min_val = spec['min']
                max_val = spec['max']
                vec[i] = (value - min_val) / (max_val - min_val)
        
        return vec
    
    def _denormalize_config(self, vec: np.ndarray) -> Dict[str, Any]:
        """Convert normalized vector to config dict."""
        
        config = {}
        
        for i, param in enumerate(self._param_names):
            spec = self._param_space[param]
            
            if spec['type'] == 'bool':
                config[param] = int(vec[i] > 0.5)
            elif spec['type'] == 'int':
                min_val = spec['min']
                max_val = spec['max']
                value = min_val + vec[i] * (max_val - min_val)
                config[param] = int(round(value))
            else:  # float
                min_val = spec['min']
                max_val = spec['max']
                config[param] = min_val + vec[i] * (max_val - min_val)
        
        return config
    
    async def suggest_config(
        self,
        context: str,
        current_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Suggest next configuration to try using Bayesian optimization.
        
        Args:
            context: Task type (code_search, debugging, etc.)
            current_config: Current configuration (optional)
        
        Returns:
            Suggested configuration
        """
        
        async with self._lock:
            # Get or create GP model for this context
            if context not in self._gp_models:
                self._gp_models[context] = GaussianProcess(self._dim)
            
            gp = self._gp_models[context]
            
            # Decide: explore or exploit
            if np.random.random() < self._exploration_rate or len(gp.X_train) < 5:
                # Exploration: random sample
                return self._random_sample()
            
            # Exploitation: optimize acquisition function
            return await self._optimize_acquisition(gp, context)
    
    def _random_sample(self) -> Dict[str, Any]:
        """Sample random configuration."""
        
        vec = np.random.random(self._dim).astype(np.float32)
        return self._denormalize_config(vec)
    
    async def _optimize_acquisition(
        self,
        gp: GaussianProcess,
        context: str
    ) -> Dict[str, Any]:
        """
        Optimize acquisition function (Expected Improvement + Thompson Sampling).
        """
        
        # Get current best
        best_y = max(gp.y_train) if gp.y_train else 0.0
        
        # Thompson Sampling: sample multiple points
        candidates = []
        ei_values = []
        
        for _ in range(self._thompson_samples):
            # Random candidate
            x = np.random.random(self._dim).astype(np.float32)
            
            # Predict
            mean, std = gp.predict(x)
            
            # Expected Improvement
            if std > 0:
                z = (mean - best_y) / std
                ei = std * (z * self._normal_cdf(z) + self._normal_pdf(z))
            else:
                ei = 0.0
            
            candidates.append(x)
            ei_values.append(ei)
        
        # Select best candidate
        best_idx = np.argmax(ei_values)
        best_x = candidates[best_idx]
        
        return self._denormalize_config(best_x)
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal."""
        return 0.5 * (1.0 + np.tanh(x / np.sqrt(2.0)))
    
    def _normal_pdf(self, x: float) -> float:
        """Probability density function for standard normal."""
        return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    
    async def record_observation(
        self,
        config: Dict[str, Any],
        context: str,
        latency_ms: float,
        success_rate: float
    ):
        """
        Record observation of config performance.
        
        Updates GP model with new data point.
        """
        
        async with self._lock:
            # Compute quality score
            latency_score = max(0, 1.0 - (latency_ms / 5000))
            quality_score = 0.6 * success_rate + 0.4 * latency_score
            
            # Create observation
            obs = ConfigObservation(
                config=config,
                context=context,
                latency_ms=latency_ms,
                success_rate=success_rate,
                quality_score=quality_score,
                timestamp=time.time()
            )
            
            self._observations.append(obs)
            
            # Update GP model
            if context not in self._gp_models:
                self._gp_models[context] = GaussianProcess(self._dim)
            
            gp = self._gp_models[context]
            
            # Normalize config
            x = self._normalize_config(config)
            
            # Add to GP
            gp.add_observation(x, quality_score)
            
            # Update best config for this context
            if context not in self._best_configs or quality_score > self._best_configs[context][1]:
                self._best_configs[context] = (config.copy(), quality_score)
            
            # Periodically retrain with only recent data to avoid overfitting
            if len(gp.X_train) > 200:
                # Keep only last 150 points
                gp.X_train = gp.X_train[-150:]
                gp.y_train = gp.y_train[-150:]
                gp._need_update = True
    
    async def get_best_config(self, context: str) -> Optional[Dict[str, Any]]:
        """Get best known configuration for context."""
        
        async with self._lock:
            if context in self._best_configs:
                return self._best_configs[context][0].copy()
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        
        return {
            'total_observations': len(self._observations),
            'contexts': list(self._gp_models.keys()),
            'gp_models': {
                context: {
                    'training_points': len(gp.X_train),
                    'best_quality': max(gp.y_train) if gp.y_train else 0.0
                }
                for context, gp in self._gp_models.items()
            },
            'best_configs': {
                context: {
                    'quality': quality,
                    'config': config
                }
                for context, (config, quality) in self._best_configs.items()
            },
            'exploration_rate': self._exploration_rate
        }


# Singleton
_bayesian_optimizer: Optional[BayesianConfigOptimizer] = None
_optimizer_lock = asyncio.Lock()


async def get_bayesian_optimizer() -> BayesianConfigOptimizer:
    """Get singleton Bayesian optimizer."""
    global _bayesian_optimizer
    if _bayesian_optimizer is None:
        async with _optimizer_lock:
            if _bayesian_optimizer is None:
                _bayesian_optimizer = BayesianConfigOptimizer()
    return _bayesian_optimizer


__all__ = [
    "BayesianConfigOptimizer",
    "get_bayesian_optimizer",
]
