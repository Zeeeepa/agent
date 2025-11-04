"""ML Orchestrator - Production integration layer for all ML components.

Integrates:
- Semantic cache
- ML classifiers
- Monitoring
- User profiles
- Explanations
- Batch processing

This is the main entry point for all ML operations.
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Tuple, Any


class MLOrchestrator:
    """
    Production ML orchestration layer.
    
    Coordinates all ML components and ensures proper integration.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Component references (lazy loaded)
        self._cache = None
        self._classifier = None
        self._monitor = None
        self._user_learner = None
        self._explainer = None
        self._batch_processor = None
    
    async def initialize(self):
        """Initialize all ML components."""
        
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Initialize cache
                from jinx.micro.embeddings.semantic_cache import get_embedding_cache
                self._cache = await get_embedding_cache()
                
                # Initialize classifier
                from jinx.micro.runtime.ml_task_classifier import get_ml_classifier
                self._classifier = await get_ml_classifier()
                
                # Initialize monitoring
                from jinx.micro.runtime.ml_monitoring import get_ml_monitoring
                self._monitor = await get_ml_monitoring()
                
                # Initialize user learner
                from jinx.micro.runtime.user_profile_learner import get_user_learner
                self._user_learner = await get_user_learner()
                
                # Initialize explainer
                from jinx.micro.runtime.prediction_explainer import get_explainer
                self._explainer = await get_explainer()
                
                # Initialize batch processor
                from jinx.micro.embeddings.batch_processor import get_batch_processor
                self._batch_processor = await get_batch_processor()
                
                self._initialized = True
            
            except Exception:
                pass
    
    async def predict_task(
        self,
        query: str,
        context: Optional[Dict] = None,
        user_id: Optional[str] = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Complete ML prediction pipeline.
        
        Args:
            query: User query
            context: Additional context
            user_id: User identifier for personalization
            explain: Generate explanation
        
        Returns:
            {
                'task_type': str,
                'confidence': float,
                'all_scores': dict,
                'latency_ms': float,
                'from_cache': bool,
                'explanation': dict (if explain=True),
                'user_adapted': bool
            }
        """
        
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # === STEP 1: Try semantic cache ===
            from_cache = False
            embedding = None
            
            if self._cache:
                cache_result = await self._cache.get(query, source='task_prediction')
                
                if cache_result:
                    embedding, from_cache = cache_result
            
            # === STEP 2: Get embedding if not cached ===
            if embedding is None:
                from jinx.micro.embeddings.pipeline import embed_text
                
                emb_obj = await embed_text(query, source='task_prediction')
                
                if emb_obj and 'embedding' in emb_obj:
                    embedding = emb_obj['embedding']
                    
                    # Cache it
                    if self._cache and embedding:
                        await self._cache.put(query, embedding, source='task_prediction')
            
            # === STEP 3: ML Classification ===
            if self._classifier and embedding:
                query_vec = np.array(embedding, dtype=np.float32)
                
                # Get base prediction
                task_type, confidence, all_scores = await self._classifier.predict(
                    query,
                    context
                )
                
                # === STEP 4: User adaptation ===
                user_adapted = False
                
                if user_id and self._user_learner:
                    adapted_task, adapted_conf = await self._user_learner.adapt_prediction(
                        user_id=user_id,
                        query_embedding=query_vec,
                        base_prediction=(task_type, confidence)
                    )
                    
                    if adapted_task != task_type or abs(adapted_conf - confidence) > 0.05:
                        task_type = adapted_task
                        confidence = adapted_conf
                        user_adapted = True
            
            else:
                # Fallback
                task_type = 'conversation'
                confidence = 0.5
                all_scores = {'conversation': 0.5}
                query_vec = None
            
            # === STEP 5: Generate explanation ===
            explanation = None
            
            if explain and self._explainer and query_vec is not None:
                explanation_obj = await self._explainer.explain(
                    query=query,
                    query_embedding=query_vec,
                    predicted_task=task_type,
                    confidence=confidence,
                    all_scores=all_scores,
                    context=context
                )
                
                explanation = {
                    'predicted_task': explanation_obj.predicted_task,
                    'confidence': explanation_obj.confidence,
                    'top_features': explanation_obj.top_features[:5],
                    'alternatives': explanation_obj.alternatives,
                    'confidence_breakdown': explanation_obj.confidence_breakdown,
                    'counterfactual': explanation_obj.counterfactual
                }
            
            # === STEP 6: Record metrics ===
            latency_ms = (time.time() - start_time) * 1000
            
            if self._monitor:
                await self._monitor.record_prediction(
                    task_type=task_type,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    success=True,
                    quality_score=confidence  # Use confidence as proxy
                )
            
            # Return result
            return {
                'task_type': task_type,
                'confidence': confidence,
                'all_scores': all_scores,
                'latency_ms': latency_ms,
                'from_cache': from_cache,
                'explanation': explanation,
                'user_adapted': user_adapted
            }
        
        except Exception as e:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            
            if self._monitor:
                await self._monitor.record_prediction(
                    task_type='unknown',
                    confidence=0.0,
                    latency_ms=latency_ms,
                    success=False,
                    quality_score=0.0
                )
            
            # Return safe fallback
            return {
                'task_type': 'conversation',
                'confidence': 0.5,
                'all_scores': {'conversation': 0.5},
                'latency_ms': latency_ms,
                'from_cache': False,
                'explanation': None,
                'user_adapted': False,
                'error': str(e)
            }
    
    async def learn_from_feedback(
        self,
        query: str,
        predicted_task: str,
        actual_task: Optional[str],
        outcome_quality: float,
        user_id: Optional[str] = None
    ):
        """
        Learn from user interaction feedback.
        
        Updates all learning components.
        """
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get embedding
            from jinx.micro.embeddings.pipeline import embed_text
            
            emb_obj = await embed_text(query, source='task_prediction')
            
            if not emb_obj or 'embedding' not in emb_obj:
                return
            
            embedding = np.array(emb_obj['embedding'], dtype=np.float32)
            
            # Update classifier
            if self._classifier:
                await self._classifier.learn_from_feedback(
                    text=query,
                    predicted_task=predicted_task,
                    actual_task=actual_task,
                    outcome_quality=outcome_quality
                )
            
            # Update user profile
            if user_id and self._user_learner:
                await self._user_learner.learn_from_interaction(
                    user_id=user_id,
                    query=query,
                    query_embedding=embedding,
                    true_task_type=actual_task or predicted_task,
                    outcome_quality=outcome_quality
                )
        
        except Exception:
            pass
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get complete system statistics."""
        
        if not self._initialized:
            await self.initialize()
        
        stats = {}
        
        try:
            if self._cache:
                stats['cache'] = self._cache.get_stats()
        except Exception:
            pass
        
        try:
            if self._monitor:
                stats['monitoring'] = self._monitor.get_metrics_summary()
        except Exception:
            pass
        
        try:
            if self._batch_processor:
                stats['batch_processor'] = self._batch_processor.get_stats()
        except Exception:
            pass
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        
        health = {
            'initialized': self._initialized,
            'cache_available': self._cache is not None,
            'classifier_available': self._classifier is not None,
            'monitor_available': self._monitor is not None,
            'user_learner_available': self._user_learner is not None,
            'explainer_available': self._explainer is not None,
            'batch_processor_available': self._batch_processor is not None
        }
        
        return health


# Singleton
_ml_orchestrator: Optional[MLOrchestrator] = None
_orchestrator_lock = asyncio.Lock()


async def get_ml_orchestrator() -> MLOrchestrator:
    """Get singleton ML orchestrator."""
    global _ml_orchestrator
    if _ml_orchestrator is None:
        async with _orchestrator_lock:
            if _ml_orchestrator is None:
                _ml_orchestrator = MLOrchestrator()
                await _ml_orchestrator.initialize()
    return _ml_orchestrator


__all__ = [
    "MLOrchestrator",
    "get_ml_orchestrator",
]
