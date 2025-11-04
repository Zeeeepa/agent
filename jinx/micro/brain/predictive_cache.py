"""Intelligent predictive caching using pattern recognition and embeddings.

Replaces primitive LRU with ML-driven prefetching and adaptive eviction.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from jinx.micro.embeddings.embed_cache import embed_text_cached as _embed
except Exception:
    _embed = None  # type: ignore


@dataclass
class CacheEntry:
    """Enhanced cache entry with ML features."""
    key: str
    value: Any
    size: int
    created: float
    last_access: float
    access_count: int
    embedding: Optional[List[float]] = None
    predicted_next_access: float = 0.0
    utility_score: float = 0.0


class PredictiveCacheManager:
    """ML-driven cache with pattern recognition and predictive prefetching."""
    
    def __init__(
        self,
        max_size_bytes: int = 20 * 1024 * 1024,
        max_entries: int = 512,
        ttl_seconds: float = 600.0
    ):
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size = 0
        
        # Access pattern tracking for prediction
        self.access_sequence: List[Tuple[str, float]] = []
        self.access_patterns: Dict[str, List[str]] = {}  # key -> commonly accessed next keys
        
        # Embedding-based similarity index for semantic prefetch
        self.embedding_index: Dict[str, List[float]] = {}
        
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with ML-enhanced lookup."""
        async with self._lock:
            entry = self.entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            now = time.time()
            if (now - entry.created) > self.ttl_seconds:
                await self._evict_entry(key)
                self._misses += 1
                return None
            
            # Update access stats
            entry.last_access = now
            entry.access_count += 1
            self._hits += 1
            
            # Move to end (MRU)
            self.entries.move_to_end(key)
            
            # Record access pattern
            self.access_sequence.append((key, now))
            if len(self.access_sequence) > 1000:
                self.access_sequence = self.access_sequence[-500:]
            
            # Trigger predictive prefetch
            asyncio.create_task(self._predict_and_prefetch(key))
            
            return entry.value
    
    async def put(self, key: str, value: Any, embedding: Optional[List[float]] = None) -> None:
        """Store value with ML metadata."""
        async with self._lock:
            now = time.time()
            
            # Estimate size
            try:
                import sys
                size = sys.getsizeof(value)
            except Exception:
                size = len(str(value))
            
            # Evict if needed
            while (
                (self.total_size + size > self.max_size_bytes or len(self.entries) >= self.max_entries)
                and self.entries
            ):
                await self._evict_lfu_weighted()
            
            # Update or create entry
            if key in self.entries:
                old_entry = self.entries[key]
                self.total_size -= old_entry.size
            
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                created=now,
                last_access=now,
                access_count=1,
                embedding=embedding
            )
            
            self.entries[key] = entry
            self.total_size += size
            
            # Store embedding for similarity search
            if embedding:
                self.embedding_index[key] = embedding
    
    async def _evict_entry(self, key: str) -> None:
        """Remove entry."""
        entry = self.entries.pop(key, None)
        if entry:
            self.total_size -= entry.size
            self.embedding_index.pop(key, None)
            self._evictions += 1
    
    async def _evict_lfu_weighted(self) -> None:
        """Evict using weighted LFU considering utility score."""
        if not self.entries:
            return
        
        now = time.time()
        
        # Compute utility scores for all entries
        scored_entries: List[Tuple[str, float]] = []
        for key, entry in self.entries.items():
            # Utility = access_frequency * recency * (1 - age_factor)
            age = now - entry.created
            recency = now - entry.last_access
            
            frequency_score = entry.access_count / max(1.0, age)
            recency_score = 1.0 / (1.0 + recency)
            age_factor = min(1.0, age / self.ttl_seconds)
            
            utility = frequency_score * recency_score * (1.0 - age_factor * 0.5)
            scored_entries.append((key, utility))
        
        # Evict lowest utility
        if scored_entries:
            victim_key, _ = min(scored_entries, key=lambda x: x[1])
            await self._evict_entry(victim_key)
    
    async def _predict_and_prefetch(self, accessed_key: str) -> None:
        """Predict next likely access and trigger prefetch."""
        # Build access patterns from sequence
        if len(self.access_sequence) < 2:
            return
        
        # Find what was accessed after this key in the past
        next_keys: List[str] = []
        for i in range(len(self.access_sequence) - 1):
            if self.access_sequence[i][0] == accessed_key:
                next_key = self.access_sequence[i + 1][0]
                if next_key != accessed_key:
                    next_keys.append(next_key)
        
        # Store pattern
        if next_keys:
            self.access_patterns[accessed_key] = next_keys[-5:]  # Keep last 5
        
        # Prefetch candidates (placeholder - actual prefetch depends on domain)
        candidates = self.access_patterns.get(accessed_key, [])
        if candidates:
            # Trigger domain-specific prefetch via event
            try:
                from jinx.micro.runtime.plugins import publish_event as _pub
                _pub("cache.prefetch_hint", {"keys": candidates[:3]})
            except Exception:
                pass
    
    async def get_similar_keys(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Find cached keys with similar embeddings."""
        if not query_embedding or not self.embedding_index:
            return []
        
        def _cosine(a: List[float], b: List[float]) -> float:
            if not a or not b:
                return 0.0
            try:
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = sum(x * x for x in a) ** 0.5
                norm_b = sum(x * x for x in b) ** 0.5
                return dot / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0
            except Exception:
                return 0.0
        
        scored: List[Tuple[str, float]] = []
        for key, emb in self.embedding_index.items():
            score = _cosine(query_embedding, emb)
            if score > 0.5:  # Threshold
                scored.append((key, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored[:top_k]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self._hits / max(1, self._hits + self._misses)
        return {
            'entries': len(self.entries),
            'size_bytes': self.total_size,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate': hit_rate,
            'patterns_learned': len(self.access_patterns)
        }


# Singleton
_cache: Optional[PredictiveCacheManager] = None
_cache_lock = asyncio.Lock()


async def get_predictive_cache() -> PredictiveCacheManager:
    """Get singleton predictive cache."""
    global _cache
    if _cache is None:
        async with _cache_lock:
            if _cache is None:
                _cache = PredictiveCacheManager()
    return _cache


__all__ = [
    "PredictiveCacheManager",
    "get_predictive_cache",
]
