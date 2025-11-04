"""Prediction Explainer - Makes ML predictions interpretable.

Features:
- Attention visualization (which dimensions matter)
- Nearest neighbor examples
- Feature importance
- Confidence breakdown
- Counterfactual explanations
"""

from __future__ import annotations

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class Explanation:
    """Explanation for a prediction."""
    predicted_task: str
    confidence: float
    
    # What influenced the prediction
    top_features: List[Tuple[int, float]]  # (dimension, importance)
    
    # Similar examples from training
    nearest_neighbors: List[Dict[str, Any]]
    
    # Alternative predictions
    alternatives: List[Tuple[str, float]]  # (task, confidence)
    
    # Confidence breakdown
    confidence_breakdown: Dict[str, float]
    
    # Counterfactual: what would change the prediction
    counterfactual: Optional[str]


class PredictionExplainer:
    """
    Makes ML predictions interpretable.
    
    Answers: WHY was this predicted?
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Store recent predictions with full context
        self._prediction_history: List[Dict] = []
    
    async def explain(
        self,
        query: str,
        query_embedding: np.ndarray,
        predicted_task: str,
        confidence: float,
        all_scores: Dict[str, float],
        context: Optional[Dict] = None
    ) -> Explanation:
        """
        Generate explanation for prediction.
        
        Args:
            query: Original query text
            query_embedding: Query embedding vector
            predicted_task: Predicted task type
            confidence: Prediction confidence
            all_scores: Scores for all task types
            context: Additional context
        
        Returns:
            Explanation object
        """
        
        async with self._lock:
            # === FEATURE IMPORTANCE ===
            top_features = self._get_top_features(query_embedding)
            
            # === NEAREST NEIGHBORS ===
            neighbors = await self._find_nearest_neighbors(
                query_embedding,
                predicted_task,
                k=3
            )
            
            # === ALTERNATIVES ===
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            alternatives = [(task, score) for task, score in sorted_scores[1:4]]
            
            # === CONFIDENCE BREAKDOWN ===
            breakdown = self._compute_confidence_breakdown(
                query_embedding,
                predicted_task,
                all_scores
            )
            
            # === COUNTERFACTUAL ===
            counterfactual = self._generate_counterfactual(
                query_embedding,
                predicted_task,
                all_scores
            )
            
            explanation = Explanation(
                predicted_task=predicted_task,
                confidence=confidence,
                top_features=top_features,
                nearest_neighbors=neighbors,
                alternatives=alternatives,
                confidence_breakdown=breakdown,
                counterfactual=counterfactual
            )
            
            # Store for future reference
            self._prediction_history.append({
                'query': query,
                'embedding': query_embedding,
                'prediction': predicted_task,
                'confidence': confidence,
                'explanation': explanation
            })
            
            return explanation
    
    def _get_top_features(
        self,
        embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get most important embedding dimensions.
        
        Uses magnitude as proxy for importance.
        """
        
        # Get absolute values (magnitude = importance)
        importances = np.abs(embedding)
        
        # Get top k dimensions
        top_indices = np.argsort(importances)[-top_k:][::-1]
        
        result = [
            (int(idx), float(importances[idx]))
            for idx in top_indices
        ]
        
        return result
    
    async def _find_nearest_neighbors(
        self,
        query_embedding: np.ndarray,
        predicted_task: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar past predictions.
        
        Helps user understand: "This is similar to when you asked X"
        """
        
        if not self._prediction_history:
            return []
        
        # Compute similarities
        similarities = []
        
        for entry in self._prediction_history:
            if entry['prediction'] == predicted_task:
                sim = self._cosine_similarity(
                    query_embedding,
                    entry['embedding']
                )
                similarities.append((sim, entry))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        neighbors = []
        for sim, entry in similarities[:k]:
            neighbors.append({
                'query': entry['query'],
                'similarity': float(sim),
                'confidence': entry['confidence']
            })
        
        return neighbors
    
    def _compute_confidence_breakdown(
        self,
        query_embedding: np.ndarray,
        predicted_task: str,
        all_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Break down confidence into interpretable components.
        """
        
        # Vector magnitude (how "strong" is the query)
        magnitude = float(np.linalg.norm(query_embedding))
        magnitude_norm = min(1.0, magnitude / 100.0)
        
        # Score margin (how much better than next best)
        sorted_scores = sorted(all_scores.values(), reverse=True)
        margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.5
        
        # Concentration (how focused on one task)
        entropy = -sum(
            p * np.log(p + 1e-10)
            for p in all_scores.values()
        )
        max_entropy = np.log(len(all_scores))
        concentration = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        
        return {
            'signal_strength': magnitude_norm,
            'score_margin': float(margin),
            'prediction_focus': float(concentration)
        }
    
    def _generate_counterfactual(
        self,
        query_embedding: np.ndarray,
        predicted_task: str,
        all_scores: Dict[str, float]
    ) -> Optional[str]:
        """
        Generate counterfactual explanation.
        
        "If X was different, it would be classified as Y instead"
        """
        
        # Find second-best task
        sorted_tasks = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_tasks) < 2:
            return None
        
        second_best_task, second_score = sorted_tasks[1]
        
        # Compute what would need to change
        score_diff = all_scores[predicted_task] - second_score
        
        if score_diff < 0.1:
            return f"With slightly different phrasing, this could be '{second_best_task}' (very close call)"
        elif score_diff < 0.3:
            return f"If the focus shifted slightly, this could be '{second_best_task}'"
        else:
            return None  # Prediction is very confident
    
    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cosine similarity."""
        
        dot = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
        
        return float(dot / (norm_x * norm_y))
    
    def format_explanation(self, explanation: Explanation) -> str:
        """Format explanation as human-readable text."""
        
        lines = []
        
        lines.append(f"Predicted: {explanation.predicted_task}")
        lines.append(f"Confidence: {explanation.confidence:.1%}")
        lines.append("")
        
        # Confidence breakdown
        lines.append("Confidence Breakdown:")
        for key, value in explanation.confidence_breakdown.items():
            lines.append(f"  - {key}: {value:.1%}")
        lines.append("")
        
        # Similar examples
        if explanation.nearest_neighbors:
            lines.append("Similar Past Queries:")
            for neighbor in explanation.nearest_neighbors:
                lines.append(f"  - \"{neighbor['query'][:50]}...\" ({neighbor['similarity']:.1%} similar)")
            lines.append("")
        
        # Alternatives
        if explanation.alternatives:
            lines.append("Alternative Classifications:")
            for task, score in explanation.alternatives:
                lines.append(f"  - {task}: {score:.1%}")
            lines.append("")
        
        # Counterfactual
        if explanation.counterfactual:
            lines.append(f"Note: {explanation.counterfactual}")
        
        return "\n".join(lines)


# Singleton
_explainer: Optional[PredictionExplainer] = None
_explainer_lock = asyncio.Lock()


async def get_explainer() -> PredictionExplainer:
    """Get singleton prediction explainer."""
    global _explainer
    if _explainer is None:
        async with _explainer_lock:
            if _explainer is None:
                _explainer = PredictionExplainer()
    return _explainer


__all__ = [
    "PredictionExplainer",
    "Explanation",
    "get_explainer",
]
