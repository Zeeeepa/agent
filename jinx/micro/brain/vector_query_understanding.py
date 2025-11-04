"""Vector-Based Query Understanding - Pure embeddings, NO KEYWORDS.

Replaces: brain/query_understanding.py

Advanced features:
- Code query detection via embeddings
- Temporal context from vector analysis
- Intent clustering
- Complexity analysis
- Language-agnostic
"""

from __future__ import annotations

import asyncio
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class QueryIntent:
    """Understood query intent."""
    is_code_query: bool
    temporal_context: Optional[str]  # 'recent', 'historical', None
    complexity_score: float
    confidence: float


class VectorQueryUnderstanding:
    """
    Pure vector-based query understanding.
    
    NO keywords, NO regex - only embeddings.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Learned query type centroids
        self._code_query_centroid: Optional[np.ndarray] = None
        self._conversation_centroid: Optional[np.ndarray] = None
        
        # Temporal centroids
        self._recent_centroid: Optional[np.ndarray] = None
        self._historical_centroid: Optional[np.ndarray] = None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize centroids from seed examples."""
        
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                from jinx.micro.embeddings.pipeline import embed_text
                
                # Initialize code query centroid
                code_seed = "function class method implementation code structure algorithm"
                code_emb = await embed_text(code_seed, source='query_understanding')
                if code_emb and hasattr(code_emb, 'embedding'):
                    self._code_query_centroid = np.array(code_emb.embedding, dtype=np.float32)
                
                # Initialize conversation centroid
                conv_seed = "question discussion explanation general information"
                conv_emb = await embed_text(conv_seed, source='query_understanding')
                if conv_emb and hasattr(conv_emb, 'embedding'):
                    self._conversation_centroid = np.array(conv_emb.embedding, dtype=np.float32)
                
                # Initialize temporal centroids
                recent_seed = "recent latest new current now today"
                recent_emb = await embed_text(recent_seed, source='query_understanding')
                if recent_emb and hasattr(recent_emb, 'embedding'):
                    self._recent_centroid = np.array(recent_emb.embedding, dtype=np.float32)
                
                historical_seed = "old previous past before earlier historical"
                hist_emb = await embed_text(historical_seed, source='query_understanding')
                if hist_emb and hasattr(hist_emb, 'embedding'):
                    self._historical_centroid = np.array(hist_emb.embedding, dtype=np.float32)
                
                self._initialized = True
            
            except Exception:
                pass
    
    async def understand_query(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> QueryIntent:
        """
        Understand query intent from embeddings only.
        
        Returns QueryIntent with classification.
        """
        
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            try:
                # Get query embedding
                from jinx.micro.embeddings.pipeline import embed_text
                
                emb_obj = await embed_text(query, source='query_understanding')
                
                if not emb_obj or not hasattr(emb_obj, 'embedding'):
                    return QueryIntent(
                        is_code_query=False,
                        temporal_context=None,
                        complexity_score=0.5,
                        confidence=0.5
                    )
                
                query_vec = np.array(emb_obj.embedding, dtype=np.float32)
                
                # === CODE QUERY DETECTION ===
                is_code_query = False
                code_confidence = 0.5
                
                if self._code_query_centroid is not None and self._conversation_centroid is not None:
                    code_sim = self._cosine_similarity(query_vec, self._code_query_centroid)
                    conv_sim = self._cosine_similarity(query_vec, self._conversation_centroid)
                    
                    is_code_query = code_sim > conv_sim
                    code_confidence = code_sim if is_code_query else (1.0 - conv_sim)
                
                # === TEMPORAL CONTEXT DETECTION ===
                temporal_context = None
                
                if self._recent_centroid is not None and self._historical_centroid is not None:
                    recent_sim = self._cosine_similarity(query_vec, self._recent_centroid)
                    hist_sim = self._cosine_similarity(query_vec, self._historical_centroid)
                    
                    threshold = 0.6
                    if recent_sim > threshold and recent_sim > hist_sim:
                        temporal_context = 'recent'
                    elif hist_sim > threshold and hist_sim > recent_sim:
                        temporal_context = 'historical'
                
                # === COMPLEXITY ANALYSIS ===
                # Higher std = more complex query (multiple aspects)
                complexity_score = float(np.std(query_vec))
                complexity_score = min(1.0, complexity_score / 0.3)  # Normalize
                
                return QueryIntent(
                    is_code_query=is_code_query,
                    temporal_context=temporal_context,
                    complexity_score=complexity_score,
                    confidence=code_confidence
                )
            
            except Exception:
                return QueryIntent(
                    is_code_query=False,
                    temporal_context=None,
                    complexity_score=0.5,
                    confidence=0.5
                )
    
    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cosine similarity between vectors."""
        dot = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
        
        return float(dot / (norm_x * norm_y))
    
    async def learn_from_example(
        self,
        query: str,
        is_code_query: bool,
        temporal_context: Optional[str],
        outcome_quality: float
    ):
        """Online learning from labeled examples."""
        
        async with self._lock:
            try:
                from jinx.micro.embeddings.pipeline import embed_text
                
                emb_obj = await embed_text(query, source='query_understanding')
                
                if not emb_obj or not hasattr(emb_obj, 'embedding'):
                    return
                
                query_vec = np.array(emb_obj.embedding, dtype=np.float32)
                
                # Update centroids (online learning)
                lr = 0.1 * outcome_quality
                
                if is_code_query and self._code_query_centroid is not None:
                    self._code_query_centroid = (
                        (1 - lr) * self._code_query_centroid + lr * query_vec
                    )
                elif not is_code_query and self._conversation_centroid is not None:
                    self._conversation_centroid = (
                        (1 - lr) * self._conversation_centroid + lr * query_vec
                    )
                
                if temporal_context == 'recent' and self._recent_centroid is not None:
                    self._recent_centroid = (
                        (1 - lr) * self._recent_centroid + lr * query_vec
                    )
                elif temporal_context == 'historical' and self._historical_centroid is not None:
                    self._historical_centroid = (
                        (1 - lr) * self._historical_centroid + lr * query_vec
                    )
            
            except Exception:
                pass


# Singleton
_query_understanding: Optional[VectorQueryUnderstanding] = None
_understanding_lock = asyncio.Lock()


async def get_query_understanding() -> VectorQueryUnderstanding:
    """Get singleton query understanding."""
    global _query_understanding
    if _query_understanding is None:
        async with _understanding_lock:
            if _query_understanding is None:
                _query_understanding = VectorQueryUnderstanding()
                await _query_understanding.initialize()
    return _query_understanding


__all__ = [
    "VectorQueryUnderstanding",
    "QueryIntent",
    "get_query_understanding",
]
