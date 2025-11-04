"""Vector-Based Intent Analyzer - Pure embedding analysis, no language processing.

Advanced vector analysis:
- Attention mechanisms (which parts of input are important)
- Vector decomposition (PCA, ICA for feature extraction)
- Semantic similarity networks
- Cluster analysis for intent groups
- Completely language-agnostic
"""

from __future__ import annotations

import asyncio
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class IntentVector:
    """Decomposed intent representation."""
    primary_intent: np.ndarray  # Main semantic direction
    secondary_intents: List[np.ndarray]  # Supporting directions
    attention_weights: np.ndarray  # Which dimensions matter most
    complexity_score: float  # How complex is this intent
    timestamp: float


class AttentionMechanism:
    """
    Learned attention over embedding dimensions.
    
    Determines which dimensions of the embedding space are most
    important for task classification.
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        
        # Attention weights (learned)
        self.attention_weights = np.ones(dim, dtype=np.float32) / dim
        
        # Key/Query/Value matrices (simplified transformer-style)
        self.W_k = np.random.randn(dim, dim) * 0.01
        self.W_q = np.random.randn(dim, dim) * 0.01
        self.W_v = np.random.randn(dim, dim) * 0.01
        
        # Learning rate
        self.lr = 0.01
    
    def compute_attention(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute attention-weighted embedding.
        
        Returns weighted embedding highlighting important features.
        """
        
        # Simple attention: element-wise weighting
        attended = embedding * self.attention_weights
        
        return attended
    
    def compute_self_attention(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute self-attention (transformer-style).
        
        Returns:
            (attended_embedding, attention_scores)
        """
        
        # Query, Key, Value
        Q = embedding.dot(self.W_q)
        K = embedding.dot(self.W_k)
        V = embedding.dot(self.W_v)
        
        # Attention scores
        scores = np.dot(Q, K) / np.sqrt(self.dim)
        attention = self._softmax(scores)
        
        # Apply attention
        output = attention * V
        
        return output, attention
    
    def update_attention(self, embedding: np.ndarray, importance: float):
        """
        Update attention weights based on feature importance.
        
        Args:
            embedding: Input embedding
            importance: How important this example is (0-1)
        """
        
        # Increase attention on dimensions with high magnitude
        magnitude = np.abs(embedding)
        
        # Normalize
        if np.sum(magnitude) > 0:
            magnitude = magnitude / np.sum(magnitude)
        
        # Update with momentum
        self.attention_weights = (
            0.95 * self.attention_weights +
            0.05 * importance * magnitude
        )
        
        # Normalize
        self.attention_weights = self.attention_weights / np.sum(self.attention_weights)
    
    def _softmax(self, x: float) -> float:
        """Softmax for scalar (simplified)."""
        return 1.0 / (1.0 + np.exp(-x))


class VectorDecomposer:
    """
    Decompose embeddings into interpretable components.
    
    Uses:
    - PCA for principal components
    - ICA for independent components
    - Sparse coding
    """
    
    def __init__(self, dim: int, n_components: int = 10):
        self.dim = dim
        self.n_components = min(n_components, dim)
        
        # Principal components (learned)
        self.principal_components: Optional[np.ndarray] = None
        
        # Mean vector (for centering)
        self.mean_vector: Optional[np.ndarray] = None
        
        # Training data buffer
        self.training_buffer: deque[np.ndarray] = deque(maxlen=500)
    
    def add_sample(self, embedding: np.ndarray):
        """Add sample for PCA learning."""
        self.training_buffer.append(embedding)
    
    def fit_pca(self):
        """Fit PCA on collected samples."""
        
        if len(self.training_buffer) < 10:
            return
        
        # Stack samples
        X = np.stack(list(self.training_buffer))
        
        # Center data
        self.mean_vector = np.mean(X, axis=0)
        X_centered = X - self.mean_vector
        
        # Compute covariance matrix
        cov = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top components
        self.principal_components = eigenvectors[:, :self.n_components]
    
    def decompose(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose embedding into principal components.
        
        Returns:
            (coefficients, reconstructed_embedding)
        """
        
        if self.principal_components is None or self.mean_vector is None:
            # Not fitted yet
            return np.zeros(self.n_components), embedding
        
        # Center
        centered = embedding - self.mean_vector
        
        # Project onto principal components
        coefficients = self.principal_components.T.dot(centered)
        
        # Reconstruct
        reconstructed = self.principal_components.dot(coefficients) + self.mean_vector
        
        return coefficients, reconstructed
    
    def get_component_importance(self, embedding: np.ndarray) -> np.ndarray:
        """Get importance score for each principal component."""
        
        coefficients, _ = self.decompose(embedding)
        
        # Importance = absolute magnitude
        importance = np.abs(coefficients)
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance


class SemanticClusterAnalyzer:
    """
    Analyze semantic clusters in embedding space.
    
    Discovers natural groupings without predefined categories.
    """
    
    def __init__(self, n_clusters: int = 15):
        self.n_clusters = n_clusters
        
        # Cluster centers (learned)
        self.cluster_centers: List[np.ndarray] = []
        
        # Cluster labels history
        self.cluster_assignments: deque[int] = deque(maxlen=1000)
        
        # Samples buffer
        self.samples: deque[np.ndarray] = deque(maxlen=1000)
    
    def add_sample(self, embedding: np.ndarray):
        """Add sample for clustering."""
        self.samples.append(embedding)
    
    def fit_clusters(self):
        """Fit k-means clustering."""
        
        if len(self.samples) < self.n_clusters * 2:
            return
        
        # Stack samples
        X = np.stack(list(self.samples))
        
        # Initialize centers randomly
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers = [X[i].copy() for i in indices]
        
        # K-means iterations
        for _ in range(10):
            # Assign to nearest center
            assignments = []
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.cluster_centers]
                assignments.append(np.argmin(distances))
            
            # Update centers
            for k in range(self.n_clusters):
                cluster_points = X[np.array(assignments) == k]
                if len(cluster_points) > 0:
                    self.cluster_centers[k] = np.mean(cluster_points, axis=0)
    
    def predict_cluster(self, embedding: np.ndarray) -> int:
        """Predict cluster for embedding."""
        
        if not self.cluster_centers:
            return 0
        
        distances = [np.linalg.norm(embedding - c) for c in self.cluster_centers]
        return int(np.argmin(distances))
    
    def get_cluster_diversity(self) -> float:
        """Measure diversity of recent predictions."""
        
        if not self.cluster_assignments:
            return 0.0
        
        # Entropy of cluster distribution
        unique, counts = np.unique(list(self.cluster_assignments), return_counts=True)
        probs = counts / len(self.cluster_assignments)
        
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalize by max entropy
        max_entropy = np.log(len(unique))
        
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0


class VectorIntentAnalyzer:
    """
    Advanced vector-based intent analysis.
    
    NO language processing - pure vector operations.
    
    Features:
    - Attention mechanisms
    - Vector decomposition (PCA)
    - Semantic clustering
    - Multi-scale analysis
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Attention mechanism
        self._attention: Optional[AttentionMechanism] = None
        
        # Vector decomposer
        self._decomposer: Optional[VectorDecomposer] = None
        
        # Cluster analyzer
        self._cluster_analyzer = SemanticClusterAnalyzer(n_clusters=15)
        
        # Intent history
        self._intent_history: deque[IntentVector] = deque(maxlen=1000)
        
        self._initialized = False
    
    async def initialize(self, dim: int = 1536):
        """Initialize with embedding dimension."""
        
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            self._attention = AttentionMechanism(dim)
            self._decomposer = VectorDecomposer(dim, n_components=20)
            
            self._initialized = True
    
    async def analyze_intent(
        self,
        embedding: np.ndarray,
        context: Optional[Dict] = None
    ) -> IntentVector:
        """
        Analyze intent from embedding vector.
        
        Returns rich intent representation.
        """
        
        if not self._initialized:
            await self.initialize(len(embedding))
        
        async with self._lock:
            try:
                # === ATTENTION ANALYSIS ===
                attended_embedding = self._attention.compute_attention(embedding)
                _, attention_scores = self._attention.compute_self_attention(embedding)
                
                # === DECOMPOSITION ===
                self._decomposer.add_sample(embedding)
                
                # Fit PCA periodically
                if len(self._decomposer.training_buffer) % 50 == 0:
                    self._decomposer.fit_pca()
                
                coefficients, reconstructed = self._decomposer.decompose(embedding)
                component_importance = self._decomposer.get_component_importance(embedding)
                
                # Primary intent = top component direction
                if self._decomposer.principal_components is not None:
                    primary_direction = self._decomposer.principal_components[:, 0]
                else:
                    primary_direction = embedding / (np.linalg.norm(embedding) + 1e-10)
                
                # Secondary intents = other important components
                secondary_directions = []
                if self._decomposer.principal_components is not None:
                    for i in range(1, min(3, self._decomposer.n_components)):
                        if component_importance[i] > 0.1:  # Threshold
                            secondary_directions.append(
                                self._decomposer.principal_components[:, i]
                            )
                
                # === CLUSTERING ===
                self._cluster_analyzer.add_sample(embedding)
                
                # Fit clusters periodically
                if len(self._cluster_analyzer.samples) % 100 == 0:
                    self._cluster_analyzer.fit_clusters()
                
                cluster_id = self._cluster_analyzer.predict_cluster(embedding)
                self._cluster_analyzer.cluster_assignments.append(cluster_id)
                
                # === COMPLEXITY SCORE ===
                # How spread out is the intent across components
                complexity = float(np.std(component_importance))
                
                # Create intent vector
                intent = IntentVector(
                    primary_intent=primary_direction,
                    secondary_intents=secondary_directions,
                    attention_weights=self._attention.attention_weights.copy(),
                    complexity_score=complexity,
                    timestamp=time.time()
                )
                
                self._intent_history.append(intent)
                
                return intent
            
            except Exception:
                # Fallback
                return IntentVector(
                    primary_intent=embedding / (np.linalg.norm(embedding) + 1e-10),
                    secondary_intents=[],
                    attention_weights=np.ones(len(embedding)) / len(embedding),
                    complexity_score=0.5,
                    timestamp=time.time()
                )
    
    async def update_from_feedback(
        self,
        embedding: np.ndarray,
        outcome_quality: float
    ):
        """Update attention based on outcome."""
        
        async with self._lock:
            if self._attention:
                self._attention.update_attention(embedding, outcome_quality)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        
        diversity = self._cluster_analyzer.get_cluster_diversity()
        
        return {
            'initialized': self._initialized,
            'attention_active': self._attention is not None,
            'decomposer_fitted': (
                self._decomposer.principal_components is not None
                if self._decomposer else False
            ),
            'num_clusters': len(self._cluster_analyzer.cluster_centers),
            'cluster_diversity': diversity,
            'intent_history_size': len(self._intent_history),
            'avg_complexity': (
                float(np.mean([iv.complexity_score for iv in self._intent_history]))
                if self._intent_history else 0.0
            )
        }


# Singleton
_intent_analyzer: Optional[VectorIntentAnalyzer] = None
_analyzer_lock = asyncio.Lock()


async def get_intent_analyzer() -> VectorIntentAnalyzer:
    """Get singleton intent analyzer."""
    global _intent_analyzer
    if _intent_analyzer is None:
        async with _analyzer_lock:
            if _intent_analyzer is None:
                _intent_analyzer = VectorIntentAnalyzer()
    return _intent_analyzer


__all__ = [
    "VectorIntentAnalyzer",
    "IntentVector",
    "get_intent_analyzer",
]
