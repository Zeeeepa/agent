"""ML-Powered Task Classifier - Advanced task type prediction using neural embeddings.

Uses:
- Semantic embeddings (OpenAI)
- Vector similarity with learned weights
- Historical training data
- Confidence calibration
- Multi-label support
"""

from __future__ import annotations

import asyncio
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import json


@dataclass
class TrainingExample:
    """Training example for task classifier."""
    text: str
    embedding: Optional[List[float]]
    true_label: str
    predicted_label: Optional[str]
    confidence: float
    outcome_quality: float  # 0-1, how well the prediction worked
    timestamp: float


class MLTaskClassifier:
    """
    Advanced task classifier using embeddings and learned similarities.
    
    Features:
    - Semantic embeddings from OpenAI
    - Learned centroid vectors per task type
    - Weighted similarity scoring
    - Online learning from feedback
    - Confidence calibration
    - Multi-label support (tasks can have multiple aspects)
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Task type centroids (learned from examples)
        self._centroids: Dict[str, np.ndarray] = {}
        
        # Learned feature weights (which dimensions are most important)
        self._feature_weights: Optional[np.ndarray] = None
        
        # Training history
        self._training_examples: deque[TrainingExample] = deque(maxlen=5000)
        
        # Performance per task type
        self._task_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Confidence calibration parameters
        self._calibration_params = {
            'temperature': 1.0,  # For confidence calibration
            'min_confidence': 0.5,
            'max_confidence': 0.95
        }
        
        # Online learning rate
        self._learning_rate = 0.1
        
        # Task types with semantic descriptions
        self._task_descriptions = {
            'code_search': 'finding and locating code, functions, classes, files, or specific implementations',
            'code_analysis': 'understanding how code works, analyzing logic, explaining algorithms, tracing execution',
            'debugging': 'fixing errors, investigating bugs, troubleshooting problems, addressing failures',
            'refactoring': 'improving code structure, optimizing, restructuring, cleaning up, enhancing quality',
            'implementation': 'writing new code, creating features, building functionality, implementing requirements',
            'testing': 'writing tests, validation, verification, checking correctness, quality assurance',
            'documentation': 'writing docs, comments, explanations, README files, API documentation',
            'planning': 'architectural design, planning approach, strategizing, deciding structure',
            'conversation': 'general discussion, questions, greetings, small talk, clarifications'
        }
        
        # Initialize centroids from descriptions (zero-shot)
        self._initialized = False
    
    async def initialize(self):
        """Initialize centroids from task descriptions."""
        
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                from jinx.micro.embeddings.pipeline import embed_text
                
                # Create initial centroids from descriptions
                for task_type, description in self._task_descriptions.items():
                    try:
                        embedding = await embed_text(description, source='task_classifier')
                        
                        if embedding and hasattr(embedding, 'embedding'):
                            vec = np.array(embedding.embedding, dtype=np.float32)
                            self._centroids[task_type] = vec
                    
                    except Exception:
                        pass
                
                # Initialize feature weights (uniform initially)
                if self._centroids:
                    dim = len(next(iter(self._centroids.values())))
                    self._feature_weights = np.ones(dim, dtype=np.float32)
                
                self._initialized = True
                
                # Try to load from storage
                await self._load_from_storage()
            
            except Exception:
                pass
    
    async def predict(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict task type with confidence using PURE EMBEDDINGS (no language).
        
        Enhanced with:
        - Embedding-based classifier
        - Vector intent analysis
        - Metric learning
        
        Returns:
            (task_type, confidence, all_scores)
        """
        
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            try:
                # Get embedding for input text
                from jinx.micro.embeddings.pipeline import embed_text
                
                embedding_obj = await embed_text(text, source='task_classifier')
                
                if not embedding_obj or not hasattr(embedding_obj, 'embedding'):
                    return ('conversation', 0.5, {})
                
                query_vec = np.array(embedding_obj.embedding, dtype=np.float32)
                
                # === LEVEL 1: Pure Embedding Classifier ===
                try:
                    from .embedding_task_classifier import get_embedding_classifier
                    
                    emb_classifier = await get_embedding_classifier()
                    emb_task, emb_conf, emb_scores = await emb_classifier.predict(text, context)
                    
                    # Use embedding classifier as primary
                    primary_task = emb_task
                    primary_conf = emb_conf
                    primary_scores = emb_scores
                
                except Exception:
                    # Fallback to centroid-based
                    scores = {}
                    
                    for task_type, centroid in self._centroids.items():
                        sim = self._weighted_cosine_similarity(query_vec, centroid)
                        
                        if context:
                            boost = self._context_boost_vector(task_type, context, query_vec)
                            sim *= (1.0 + boost)
                        
                        scores[task_type] = float(sim)
                    
                    scores = self._softmax(scores)
                    best_task = max(scores.items(), key=lambda x: x[1])
                    primary_task, primary_conf = best_task
                    primary_scores = scores
                
                # === LEVEL 2: Vector Intent Analysis ===
                try:
                    from .vector_intent_analyzer import get_intent_analyzer
                    
                    intent_analyzer = await get_intent_analyzer()
                    intent = await intent_analyzer.analyze_intent(query_vec, context)
                    
                    # Use intent complexity to adjust confidence
                    # High complexity = less confident (intent is mixed)
                    complexity_penalty = intent.complexity_score * 0.2
                    adjusted_conf = primary_conf * (1.0 - complexity_penalty)
                    
                    primary_conf = max(0.5, min(0.95, adjusted_conf))
                
                except Exception:
                    pass
                
                # Final calibration based on history
                calibrated_confidence = self._calibrate_confidence(
                    primary_task,
                    primary_conf
                )
                
                return (primary_task, calibrated_confidence, primary_scores)
            
            except Exception:
                return ('conversation', 0.5, {})
    
    def _weighted_cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute weighted cosine similarity."""
        
        if self._feature_weights is None:
            # Unweighted
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot / (norm1 * norm2)
        
        # Weighted
        weighted_vec1 = vec1 * self._feature_weights
        weighted_vec2 = vec2 * self._feature_weights
        
        dot = np.dot(weighted_vec1, weighted_vec2)
        norm1 = np.linalg.norm(weighted_vec1)
        norm2 = np.linalg.norm(weighted_vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to probabilities using softmax."""
        
        if not scores:
            return {}
        
        # Temperature scaling
        temp = self._calibration_params['temperature']
        
        values = np.array(list(scores.values()), dtype=np.float32)
        values = values / temp
        
        # Numerical stability
        values = values - np.max(values)
        
        exp_values = np.exp(values)
        sum_exp = np.sum(exp_values)
        
        if sum_exp == 0:
            # Uniform distribution
            n = len(scores)
            return {k: 1.0/n for k in scores.keys()}
        
        probs = exp_values / sum_exp
        
        return {k: float(p) for k, p in zip(scores.keys(), probs)}
    
    def _calibrate_confidence(
        self,
        task_type: str,
        raw_confidence: float
    ) -> float:
        """Calibrate confidence based on historical performance."""
        
        # Get historical accuracy for this task type
        history = self._task_performance.get(task_type, [])
        
        if not history:
            # No history - conservative confidence
            return max(
                self._calibration_params['min_confidence'],
                raw_confidence * 0.8
            )
        
        # Average historical performance
        avg_performance = np.mean(history[-50:])  # Last 50 predictions
        
        # Calibrate: if we're often wrong, reduce confidence
        calibrated = raw_confidence * (0.5 + 0.5 * avg_performance)
        
        # Clamp
        return max(
            self._calibration_params['min_confidence'],
            min(self._calibration_params['max_confidence'], calibrated)
        )
    
    def _context_boost_vector(
        self,
        task_type: str,
        context: Dict,
        query_vec: np.ndarray
    ) -> float:
        """Vector-based context boosting (language-agnostic)."""
        
        boost = 0.0
        
        # Convert context to vector features
        has_error = float(context.get('recent_error', False))
        edit_count = float(context.get('recent_edits', 0))
        transcript_size = float(context.get('transcript_size', 0))
        
        # Normalize features
        edit_norm = min(1.0, edit_count / 10.0)
        transcript_norm = min(1.0, transcript_size / 5000.0)
        
        # Create context vector
        context_vec = np.array([has_error, edit_norm, transcript_norm], dtype=np.float32)
        
        # Boost based on vector magnitude and query embedding
        # High-dimensional tasks (complex) benefit from more context
        query_complexity = np.std(query_vec)  # Spread of embedding
        context_strength = np.linalg.norm(context_vec)
        
        boost = float(query_complexity * context_strength * 0.3)
        
        return boost
    
    async def learn_from_feedback(
        self,
        text: str,
        predicted_task: str,
        actual_task: Optional[str],
        outcome_quality: float
    ):
        """
        Online learning from feedback - updates ALL classifiers.
        
        Args:
            text: Original query
            predicted_task: What we predicted
            actual_task: Ground truth (if known)
            outcome_quality: How well the prediction worked (0-1)
        """
        
        async with self._lock:
            try:
                # Get embedding
                from jinx.micro.embeddings.pipeline import embed_text
                
                embedding_obj = await embed_text(text, source='task_classifier')
                
                if not embedding_obj or not hasattr(embedding_obj, 'embedding'):
                    return
                
                query_vec = np.array(embedding_obj.embedding, dtype=np.float32)
                
                # Record performance
                self._task_performance[predicted_task].append(outcome_quality)
                
                # Store training example
                example = TrainingExample(
                    text=text,
                    embedding=query_vec.tolist(),
                    true_label=actual_task or predicted_task,
                    predicted_label=predicted_task,
                    confidence=0.0,
                    outcome_quality=outcome_quality,
                    timestamp=time.time()
                )
                
                self._training_examples.append(example)
                
                # === UPDATE CENTROID CLASSIFIER ===
                if actual_task and actual_task in self._centroids:
                    centroid = self._centroids[actual_task]
                    lr = self._learning_rate * outcome_quality
                    self._centroids[actual_task] = (
                        (1 - lr) * centroid + lr * query_vec
                    )
                
                # === UPDATE EMBEDDING CLASSIFIER ===
                try:
                    from .embedding_task_classifier import get_embedding_classifier
                    
                    emb_classifier = await get_embedding_classifier()
                    await emb_classifier.learn_from_feedback(
                        text=text,
                        predicted_task=predicted_task,
                        outcome_quality=outcome_quality,
                        actual_task=actual_task
                    )
                except Exception:
                    pass
                
                # === UPDATE INTENT ANALYZER ===
                try:
                    from .vector_intent_analyzer import get_intent_analyzer
                    
                    intent_analyzer = await get_intent_analyzer()
                    await intent_analyzer.update_from_feedback(
                        embedding=query_vec,
                        outcome_quality=outcome_quality
                    )
                except Exception:
                    pass
                
                # Update feature weights
                await self._update_feature_weights()
                
                # Periodically save
                if len(self._training_examples) % 100 == 0:
                    await self._save_to_storage()
            
            except Exception:
                pass
    
    async def _update_feature_weights(self):
        """Update feature importance weights using variance analysis."""
        
        if len(self._training_examples) < 50:
            return
        
        try:
            # Get recent high-quality examples
            high_quality = [
                ex for ex in list(self._training_examples)[-200:]
                if ex.outcome_quality > 0.7 and ex.embedding
            ]
            
            if len(high_quality) < 20:
                return
            
            # Group by task type
            by_task = defaultdict(list)
            for ex in high_quality:
                by_task[ex.true_label].append(np.array(ex.embedding))
            
            # Compute within-class variance vs between-class variance
            # Features with high between/within ratio are more important
            
            if self._feature_weights is not None:
                dim = len(self._feature_weights)
                
                # Compute global mean
                all_vecs = [np.array(ex.embedding) for ex in high_quality]
                global_mean = np.mean(all_vecs, axis=0)
                
                # Between-class variance
                between_var = np.zeros(dim)
                for task_type, vecs in by_task.items():
                    task_mean = np.mean(vecs, axis=0)
                    diff = task_mean - global_mean
                    between_var += len(vecs) * (diff ** 2)
                
                # Within-class variance
                within_var = np.zeros(dim)
                for task_type, vecs in by_task.items():
                    task_mean = np.mean(vecs, axis=0)
                    for vec in vecs:
                        diff = vec - task_mean
                        within_var += diff ** 2
                
                # Fisher ratio (avoid division by zero)
                ratio = between_var / (within_var + 1e-8)
                
                # Normalize to [0.5, 1.5] range
                ratio = np.clip(ratio, 0, np.percentile(ratio, 95))
                ratio = 0.5 + (ratio / (np.max(ratio) + 1e-8))
                
                # Smooth update
                self._feature_weights = (
                    0.9 * self._feature_weights + 0.1 * ratio
                )
        
        except Exception:
            pass
    
    async def _save_to_storage(self):
        """Save learned parameters to disk."""
        
        try:
            import os
            
            storage_dir = '.jinx/ml'
            os.makedirs(storage_dir, exist_ok=True)
            
            # Save centroids
            centroids_data = {
                task: centroid.tolist()
                for task, centroid in self._centroids.items()
            }
            
            with open(f'{storage_dir}/centroids.json', 'w') as f:
                json.dump(centroids_data, f)
            
            # Save feature weights
            if self._feature_weights is not None:
                np.save(f'{storage_dir}/feature_weights.npy', self._feature_weights)
            
            # Save calibration params
            with open(f'{storage_dir}/calibration.json', 'w') as f:
                json.dump(self._calibration_params, f)
        
        except Exception:
            pass
    
    async def _load_from_storage(self):
        """Load learned parameters from disk."""
        
        try:
            import os
            
            storage_dir = '.jinx/ml'
            
            # Load centroids
            centroids_path = f'{storage_dir}/centroids.json'
            if os.path.exists(centroids_path):
                with open(centroids_path, 'r') as f:
                    centroids_data = json.load(f)
                
                self._centroids = {
                    task: np.array(vec, dtype=np.float32)
                    for task, vec in centroids_data.items()
                }
            
            # Load feature weights
            weights_path = f'{storage_dir}/feature_weights.npy'
            if os.path.exists(weights_path):
                self._feature_weights = np.load(weights_path)
            
            # Load calibration
            calibration_path = f'{storage_dir}/calibration.json'
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r') as f:
                    self._calibration_params.update(json.load(f))
        
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        
        return {
            'initialized': self._initialized,
            'num_centroids': len(self._centroids),
            'feature_weights_set': self._feature_weights is not None,
            'training_examples': len(self._training_examples),
            'task_performance': {
                task: {
                    'samples': len(perf),
                    'avg_quality': float(np.mean(perf[-50:])) if perf else 0.0
                }
                for task, perf in self._task_performance.items()
            },
            'calibration': self._calibration_params
        }


# Singleton
_ml_classifier: Optional[MLTaskClassifier] = None
_classifier_lock = asyncio.Lock()


async def get_ml_classifier() -> MLTaskClassifier:
    """Get singleton ML classifier."""
    global _ml_classifier
    if _ml_classifier is None:
        async with _classifier_lock:
            if _ml_classifier is None:
                _ml_classifier = MLTaskClassifier()
                await _ml_classifier.initialize()
    return _ml_classifier


__all__ = [
    "MLTaskClassifier",
    "get_ml_classifier",
]
