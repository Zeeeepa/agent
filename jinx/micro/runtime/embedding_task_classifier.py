"""Pure Embedding-Based Task Classifier - Language-agnostic, no keywords/regex.

Advanced ML approaches:
- Contrastive learning (SimCLR-style)
- Metric learning (learnable distance functions)
- Prototype networks (few-shot learning)
- Meta-learning (MAML-style adaptation)
- Completely language-agnostic - works on pure vector representations
"""

from __future__ import annotations

import asyncio
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, defaultdict


@dataclass
class TaskPrototype:
    """Learned prototype (centroid) for a task type."""
    task_type: str
    embedding: np.ndarray
    support_count: int  # Number of examples
    confidence: float  # Quality of this prototype
    last_updated: float


class MetricLearner:
    """
    Learnable distance metric using Mahalanobis distance.
    
    Instead of euclidean distance, learn a matrix M such that:
    d(x, y) = sqrt((x-y)^T M (x-y))
    
    This allows the model to learn which dimensions are important.
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        
        # Initialize M as identity (euclidean distance)
        self.M = np.eye(dim, dtype=np.float32)
        
        # Learning rate
        self.lr = 0.01
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute learned distance between x and y."""
        diff = x - y
        
        # Mahalanobis distance: sqrt(diff^T M diff)
        dist_sq = diff.dot(self.M).dot(diff)
        
        return float(np.sqrt(max(0, dist_sq)))
    
    def update_metric(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray
    ):
        """
        Update metric using triplet loss.
        
        Goal: distance(anchor, positive) < distance(anchor, negative)
        """
        
        # Compute distances
        d_pos = self.distance(anchor, positive)
        d_neg = self.distance(anchor, negative)
        
        # Triplet loss with margin
        margin = 0.5
        loss = max(0, d_pos - d_neg + margin)
        
        if loss > 0:
            # Gradient descent on M
            diff_pos = anchor - positive
            diff_neg = anchor - negative
            
            # Update M to increase distance to negative, decrease to positive
            gradient = np.outer(diff_neg, diff_neg) - np.outer(diff_pos, diff_pos)
            
            self.M += self.lr * gradient
            
            # Keep M positive semi-definite (project to valid space)
            # Simple approach: ensure diagonal is positive
            np.fill_diagonal(self.M, np.maximum(np.diag(self.M), 0.01))


class ContrastiveLearner:
    """
    Contrastive learning for task embeddings.
    
    Learns to pull similar tasks together and push different tasks apart.
    Uses SimCLR-style NT-Xent loss.
    """
    
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
        
        # Store positive pairs (same task)
        self.positive_pairs: deque[Tuple[np.ndarray, np.ndarray, str]] = deque(maxlen=1000)
        
        # Store negative pairs (different tasks)
        self.negative_pairs: deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=1000)
    
    def add_example(self, embedding: np.ndarray, task_type: str):
        """Add example for contrastive learning."""
        
        # Find similar examples (same task) for positive pairs
        for other_emb, other_task in [(p[0], p[2]) for p in self.positive_pairs]:
            if other_task == task_type:
                # Positive pair
                self.positive_pairs.append((embedding, other_emb, task_type))
            else:
                # Negative pair
                self.negative_pairs.append((embedding, other_emb))
        
        # Store this example
        self.positive_pairs.append((embedding, embedding, task_type))
    
    def compute_contrastive_loss(self) -> float:
        """
        Compute NT-Xent (normalized temperature-scaled cross entropy) loss.
        
        Lower loss = better separation between task types.
        """
        
        if len(self.positive_pairs) < 2:
            return 0.0
        
        # Sample batch
        batch_size = min(32, len(self.positive_pairs))
        indices = np.random.choice(len(self.positive_pairs), batch_size, replace=False)
        
        batch_pos = [self.positive_pairs[i] for i in indices]
        
        total_loss = 0.0
        
        for anchor, positive, task in batch_pos:
            # Similarity to positive
            sim_pos = self._cosine_similarity(anchor, positive) / self.temperature
            
            # Similarities to negatives
            sims_neg = []
            for neg_pair in list(self.negative_pairs)[-50:]:  # Last 50 negatives
                sim_neg = self._cosine_similarity(anchor, neg_pair[1]) / self.temperature
                sims_neg.append(sim_neg)
            
            if sims_neg:
                # NT-Xent loss
                exp_pos = np.exp(sim_pos)
                sum_exp = exp_pos + np.sum(np.exp(sims_neg))
                
                loss = -np.log(exp_pos / sum_exp)
                total_loss += loss
        
        return total_loss / batch_size
    
    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cosine similarity between vectors."""
        dot = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
        
        return float(dot / (norm_x * norm_y))


class EmbeddingTaskClassifier:
    """
    Pure embedding-based task classifier.
    
    NO keywords, NO regex, NO language-specific features.
    Works entirely on semantic embeddings.
    
    Features:
    - Prototype networks (learned centroids)
    - Metric learning (Mahalanobis distance)
    - Contrastive learning (SimCLR)
    - Few-shot adaptation
    - Meta-learning ready
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Task prototypes (learned centroids)
        self._prototypes: Dict[str, TaskPrototype] = {}
        
        # Metric learner
        self._metric_learner: Optional[MetricLearner] = None
        
        # Contrastive learner
        self._contrastive_learner = ContrastiveLearner()
        
        # Training examples (for few-shot learning)
        self._examples: deque[Tuple[np.ndarray, str, float]] = deque(maxlen=5000)
        
        # Task types (defined by initial examples, not keywords)
        self._task_types = [
            'code_search', 'code_analysis', 'debugging', 'refactoring',
            'implementation', 'testing', 'documentation', 'planning', 'conversation'
        ]
        
        # Performance tracking per task
        self._task_quality: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))
        
        # Adaptive parameters
        self._confidence_threshold = 0.6
        self._similarity_threshold = 0.7
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize prototypes from seed examples."""
        
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Initialize prototypes using seed descriptions
                seed_descriptions = await self._get_seed_descriptions()
                
                for task_type, description in seed_descriptions.items():
                    embedding = await self._get_embedding(description)
                    
                    if embedding is not None:
                        self._prototypes[task_type] = TaskPrototype(
                            task_type=task_type,
                            embedding=embedding,
                            support_count=1,
                            confidence=0.5,
                            last_updated=time.time()
                        )
                
                # Initialize metric learner
                if self._prototypes:
                    first_proto = next(iter(self._prototypes.values()))
                    dim = len(first_proto.embedding)
                    self._metric_learner = MetricLearner(dim)
                
                self._initialized = True
                
                # Try to load from storage
                await self._load_from_storage()
            
            except Exception:
                pass
    
    async def _get_seed_descriptions(self) -> Dict[str, str]:
        """Get seed descriptions for initialization (language-agnostic concepts)."""
        
        # These are ONLY for initialization - will be replaced by learned prototypes
        return {
            'code_search': 'locating specific code patterns functions classes implementations',
            'code_analysis': 'understanding code logic flow algorithms execution tracing',
            'debugging': 'investigating errors failures exceptions problems issues',
            'refactoring': 'improving code structure quality organization optimization',
            'implementation': 'creating new functionality features code building',
            'testing': 'verification validation quality assurance checking',
            'documentation': 'explaining describing documenting code purpose',
            'planning': 'designing architecture structure strategy approach',
            'conversation': 'general discussion questions clarifications dialogue'
        }
    
    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text."""
        
        try:
            from jinx.micro.embeddings.pipeline import embed_text
            
            result = await embed_text(text, source='task_classifier')
            
            if result and hasattr(result, 'embedding'):
                return np.array(result.embedding, dtype=np.float32)
        
        except Exception:
            pass
        
        return None
    
    async def predict(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict task type from embedding only.
        
        Returns:
            (task_type, confidence, all_scores)
        """
        
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            try:
                # Get embedding for query
                query_embedding = await self._get_embedding(text)
                
                if query_embedding is None:
                    return ('conversation', 0.5, {})
                
                # Compute distances to all prototypes
                distances = {}
                
                for task_type, prototype in self._prototypes.items():
                    # Use learned metric
                    if self._metric_learner:
                        dist = self._metric_learner.distance(
                            query_embedding,
                            prototype.embedding
                        )
                    else:
                        # Fallback to cosine distance
                        dist = 1.0 - self._cosine_similarity(
                            query_embedding,
                            prototype.embedding
                        )
                    
                    # Weight by prototype confidence
                    weighted_dist = dist / prototype.confidence
                    
                    distances[task_type] = weighted_dist
                
                # Convert distances to similarities (lower distance = higher similarity)
                max_dist = max(distances.values()) if distances else 1.0
                similarities = {
                    task: 1.0 - (dist / max_dist)
                    for task, dist in distances.items()
                }
                
                # Softmax to get probabilities
                scores = self._softmax(similarities)
                
                # Best prediction
                best_task, best_score = max(scores.items(), key=lambda x: x[1])
                
                # Calibrate confidence based on:
                # 1. Score magnitude
                # 2. Separation from second-best
                # 3. Historical performance
                sorted_scores = sorted(scores.values(), reverse=True)
                separation = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.5
                
                confidence = best_score * (0.5 + 0.5 * separation)
                
                # Adjust by historical performance
                history = self._task_quality[best_task]
                if history:
                    avg_quality = np.mean(list(history))
                    confidence *= (0.5 + 0.5 * avg_quality)
                
                # Context boosting (vector-based features only)
                if context:
                    confidence *= (1.0 + self._context_boost_vector(query_embedding, context))
                
                confidence = max(0.5, min(0.95, confidence))
                
                return (best_task, confidence, scores)
            
            except Exception:
                return ('conversation', 0.5, {})
    
    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cosine similarity between vectors."""
        dot = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
        
        return float(dot / (norm_x * norm_y))
    
    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to probabilities."""
        
        if not scores:
            return {}
        
        values = np.array(list(scores.values()), dtype=np.float32)
        
        # Temperature scaling
        temperature = 0.5
        values = values / temperature
        
        # Stability
        values = values - np.max(values)
        
        exp_values = np.exp(values)
        sum_exp = np.sum(exp_values)
        
        if sum_exp == 0:
            n = len(scores)
            return {k: 1.0/n for k in scores.keys()}
        
        probs = exp_values / sum_exp
        
        return {k: float(p) for k, p in zip(scores.keys(), probs)}
    
    def _context_boost_vector(self, embedding: np.ndarray, context: Dict) -> float:
        """Context boosting using vector features only (no language)."""
        
        boost = 0.0
        
        # Numerical features from context
        has_error = float(context.get('recent_error', False))
        edit_count = float(context.get('recent_edits', 0))
        transcript_size = float(context.get('transcript_size', 0))
        
        # Normalize
        edit_count_norm = min(1.0, edit_count / 10.0)
        transcript_size_norm = min(1.0, transcript_size / 5000.0)
        
        # Combine features
        context_vec = np.array([has_error, edit_count_norm, transcript_size_norm], dtype=np.float32)
        
        # Simple boosting based on magnitude
        boost = float(np.linalg.norm(context_vec)) * 0.2
        
        return boost
    
    async def learn_from_feedback(
        self,
        text: str,
        predicted_task: str,
        outcome_quality: float,
        actual_task: Optional[str] = None
    ):
        """
        Learn from feedback using multiple techniques.
        
        Updates:
        - Prototypes (moving towards correct examples)
        - Metric learner (triplet loss)
        - Contrastive learner (positive/negative pairs)
        """
        
        async with self._lock:
            try:
                # Get embedding
                embedding = await self._get_embedding(text)
                
                if embedding is None:
                    return
                
                # Record quality
                self._task_quality[predicted_task].append(outcome_quality)
                
                # Store example
                true_task = actual_task or predicted_task
                self._examples.append((embedding, true_task, outcome_quality))
                
                # === UPDATE PROTOTYPE ===
                if true_task in self._prototypes:
                    prototype = self._prototypes[true_task]
                    
                    # Online update: move prototype towards this example
                    # Learning rate weighted by quality
                    lr = 0.1 * outcome_quality
                    
                    new_embedding = (1 - lr) * prototype.embedding + lr * embedding
                    
                    self._prototypes[true_task] = TaskPrototype(
                        task_type=true_task,
                        embedding=new_embedding,
                        support_count=prototype.support_count + 1,
                        confidence=0.9 * prototype.confidence + 0.1 * outcome_quality,
                        last_updated=time.time()
                    )
                
                # === UPDATE CONTRASTIVE LEARNER ===
                self._contrastive_learner.add_example(embedding, true_task)
                
                # === UPDATE METRIC LEARNER ===
                if self._metric_learner and len(self._examples) >= 3:
                    # Sample triplet: anchor, positive (same task), negative (different task)
                    anchor = embedding
                    anchor_task = true_task
                    
                    # Find positive (same task)
                    positives = [e for e, t, q in self._examples if t == anchor_task and q > 0.6]
                    # Find negative (different task)
                    negatives = [e for e, t, q in self._examples if t != anchor_task]
                    
                    if positives and negatives:
                        positive = positives[np.random.randint(len(positives))]
                        negative = negatives[np.random.randint(len(negatives))]
                        
                        # Update metric
                        self._metric_learner.update_metric(anchor, positive, negative)
                
                # Periodically save
                if len(self._examples) % 50 == 0:
                    await self._save_to_storage()
            
            except Exception:
                pass
    
    async def _save_to_storage(self):
        """Save learned prototypes and metrics."""
        
        try:
            import os
            import json
            
            storage_dir = '.jinx/ml_embedding'
            os.makedirs(storage_dir, exist_ok=True)
            
            # Save prototypes
            prototypes_data = {}
            for task, proto in self._prototypes.items():
                prototypes_data[task] = {
                    'embedding': proto.embedding.tolist(),
                    'support_count': proto.support_count,
                    'confidence': proto.confidence,
                    'last_updated': proto.last_updated
                }
            
            with open(f'{storage_dir}/prototypes.json', 'w') as f:
                json.dump(prototypes_data, f)
            
            # Save metric learner
            if self._metric_learner:
                np.save(f'{storage_dir}/metric_matrix.npy', self._metric_learner.M)
        
        except Exception:
            pass
    
    async def _load_from_storage(self):
        """Load learned prototypes and metrics."""
        
        try:
            import os
            import json
            
            storage_dir = '.jinx/ml_embedding'
            
            # Load prototypes
            proto_path = f'{storage_dir}/prototypes.json'
            if os.path.exists(proto_path):
                with open(proto_path, 'r') as f:
                    prototypes_data = json.load(f)
                
                for task, data in prototypes_data.items():
                    self._prototypes[task] = TaskPrototype(
                        task_type=task,
                        embedding=np.array(data['embedding'], dtype=np.float32),
                        support_count=data['support_count'],
                        confidence=data['confidence'],
                        last_updated=data['last_updated']
                    )
            
            # Load metric learner
            metric_path = f'{storage_dir}/metric_matrix.npy'
            if os.path.exists(metric_path) and self._metric_learner:
                self._metric_learner.M = np.load(metric_path)
        
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        
        contrastive_loss = self._contrastive_learner.compute_contrastive_loss()
        
        return {
            'initialized': self._initialized,
            'num_prototypes': len(self._prototypes),
            'num_examples': len(self._examples),
            'metric_learner_active': self._metric_learner is not None,
            'contrastive_loss': float(contrastive_loss),
            'task_quality': {
                task: {
                    'samples': len(qualities),
                    'avg_quality': float(np.mean(list(qualities))) if qualities else 0.0
                }
                for task, qualities in self._task_quality.items()
            },
            'prototypes': {
                task: {
                    'support_count': proto.support_count,
                    'confidence': proto.confidence
                }
                for task, proto in self._prototypes.items()
            }
        }


# Singleton
_embedding_classifier: Optional[EmbeddingTaskClassifier] = None
_classifier_lock = asyncio.Lock()


async def get_embedding_classifier() -> EmbeddingTaskClassifier:
    """Get singleton embedding classifier."""
    global _embedding_classifier
    if _embedding_classifier is None:
        async with _classifier_lock:
            if _embedding_classifier is None:
                _embedding_classifier = EmbeddingTaskClassifier()
                await _embedding_classifier.initialize()
    return _embedding_classifier


__all__ = [
    "EmbeddingTaskClassifier",
    "get_embedding_classifier",
]
