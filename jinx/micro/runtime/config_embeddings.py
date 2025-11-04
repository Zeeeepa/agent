"""Configuration Embeddings - semantic understanding of configurations.

Uses embeddings to:
- Find similar successful configurations
- Understand workload patterns
- Predict optimal parameters
- Learn from historical performance
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConfigurationEmbedding:
    """Configuration with its embedding."""
    config: Dict[str, str]
    embedding: List[float]
    quality_score: float
    workload_type: str
    usage_count: int
    
    def similarity_to(self, other_embedding: List[float]) -> float:
        """Compute cosine similarity."""
        if not self.embedding or not other_embedding:
            return 0.0
        
        if len(self.embedding) != len(other_embedding):
            return 0.0
        
        # Cosine similarity
        dot = sum(a * b for a, b in zip(self.embedding, other_embedding))
        norm_a = sum(a * a for a in self.embedding) ** 0.5
        norm_b = sum(b * b for b in other_embedding) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


class ConfigEmbeddingStore:
    """Store and retrieve configurations using embeddings."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._embeddings: List[ConfigurationEmbedding] = []
        
        # Cache
        self._embedding_cache: Dict[str, List[float]] = {}
    
    async def store_config(
        self,
        config: Dict[str, str],
        quality_score: float,
        workload_type: str
    ):
        """Store configuration with its embedding."""
        
        # Create description
        desc = self._config_to_description(config, workload_type)
        
        # Get embedding
        embedding = await self._embed_text(desc)
        
        if not embedding:
            return
        
        async with self._lock:
            # Check if similar config exists
            for existing in self._embeddings:
                similarity = existing.similarity_to(embedding)
                
                if similarity > 0.95:  # Very similar
                    # Update existing
                    if quality_score > existing.quality_score:
                        existing.config = config
                        existing.quality_score = quality_score
                    existing.usage_count += 1
                    return
            
            # Add new
            self._embeddings.append(ConfigurationEmbedding(
                config=config,
                embedding=embedding,
                quality_score=quality_score,
                workload_type=workload_type,
                usage_count=1
            ))
    
    async def find_similar_configs(
        self,
        query_text: str,
        k: int = 5,
        min_similarity: float = 0.7
    ) -> List[Tuple[ConfigurationEmbedding, float]]:
        """Find similar configurations."""
        
        # Embed query
        query_embedding = await self._embed_text(query_text)
        
        if not query_embedding:
            return []
        
        # Compute similarities
        similarities = []
        
        async with self._lock:
            for config_emb in self._embeddings:
                similarity = config_emb.similarity_to(query_embedding)
                
                if similarity >= min_similarity:
                    similarities.append((config_emb, similarity))
        
        # Sort by similarity * quality
        similarities.sort(
            key=lambda x: x[1] * x[0].quality_score,
            reverse=True
        )
        
        return similarities[:k]
    
    async def find_optimal_for_workload(
        self,
        workload_type: str
    ) -> Optional[ConfigurationEmbedding]:
        """Find optimal config for workload type."""
        
        async with self._lock:
            # Filter by workload
            candidates = [
                emb for emb in self._embeddings
                if emb.workload_type == workload_type
            ]
            
            if not candidates:
                return None
            
            # Return best quality
            return max(candidates, key=lambda x: x.quality_score)
    
    async def predict_optimal_config(
        self,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Predict optimal configuration for context."""
        
        # Create context description
        workload = context.get('type', 'default')
        has_code = context.get('code_context', False)
        concurrent = context.get('concurrent_tasks', 1)
        
        desc = (
            f"workload={workload} "
            f"code_context={has_code} "
            f"concurrent={concurrent}"
        )
        
        # Find similar configs
        similar = await self.find_similar_configs(desc, k=3)
        
        if not similar:
            return None
        
        # Return config with highest combined score
        best = max(similar, key=lambda x: x[1] * x[0].quality_score)
        
        return best[0].config
    
    def _config_to_description(
        self,
        config: Dict[str, str],
        workload: str
    ) -> str:
        """Convert config to description for embedding."""
        
        parts = [f"workload={workload}"]
        
        # Extract key parameters
        for key, value in config.items():
            if 'CONC' in key or 'MAX' in key or 'MS' in key:
                parts.append(f"{key}={value}")
        
        return " ".join(parts)
    
    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        
        # Check cache
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            from jinx.micro.embeddings import embed_text_cached
            
            embedding = await embed_text_cached(text[:200])
            
            if embedding:
                self._embedding_cache[text] = embedding
            
            return embedding
        
        except Exception:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'total_configs': len(self._embeddings),
            'by_workload': self._count_by_workload(),
            'avg_quality': self._avg_quality(),
            'cache_size': len(self._embedding_cache)
        }
    
    def _count_by_workload(self) -> Dict[str, int]:
        """Count configs by workload."""
        counts = {}
        
        for emb in self._embeddings:
            workload = emb.workload_type
            counts[workload] = counts.get(workload, 0) + 1
        
        return counts
    
    def _avg_quality(self) -> float:
        """Average quality score."""
        if not self._embeddings:
            return 0.0
        
        return sum(e.quality_score for e in self._embeddings) / len(self._embeddings)


# Singleton
_store: Optional[ConfigEmbeddingStore] = None
_store_lock = asyncio.Lock()


async def get_config_embedding_store() -> ConfigEmbeddingStore:
    """Get singleton store."""
    global _store
    if _store is None:
        async with _store_lock:
            if _store is None:
                _store = ConfigEmbeddingStore()
    return _store


async def store_successful_config(
    config: Dict[str, str],
    quality_score: float,
    workload_type: str
):
    """Store successful configuration."""
    store = await get_config_embedding_store()
    await store.store_config(config, quality_score, workload_type)


async def predict_config_for_context(
    context: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """Predict optimal configuration for context."""
    store = await get_config_embedding_store()
    return await store.predict_optimal_config(context)


__all__ = [
    "ConfigEmbeddingStore",
    "ConfigurationEmbedding",
    "get_config_embedding_store",
    "store_successful_config",
    "predict_config_for_context",
]
