"""Advanced retrieval optimization with caching and adaptive strategies.

This module provides intelligent retrieval optimizations:
- Query result caching with TTL
- Adaptive k selection based on query characteristics
- Parallel retrieval from multiple sources
- Smart filtering and ranking
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import OrderedDict
import threading


@dataclass
class RetrievalMetrics:
    """Track retrieval performance metrics."""
    queries_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    total_items_retrieved: int = 0
    
    def update_latency(self, latency_ms: float) -> None:
        """Update average latency with new measurement."""
        if self.queries_count == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (
                (self.avg_latency_ms * self.queries_count + latency_ms) 
                / (self.queries_count + 1)
            )
        self.queries_count += 1


@dataclass
class CachedResult:
    """Cached retrieval result with TTL."""
    results: List[Tuple[float, str, Dict[str, Any]]]
    timestamp: float
    query_hash: str
    hit_count: int = 0


class RetrievalCache:
    """Thread-safe LRU cache with TTL for retrieval results."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: float = 300.0):
        self._cache: OrderedDict[str, CachedResult] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._metrics = RetrievalMetrics()
    
    def _compute_hash(self, query: str, k: int, filters: Optional[Dict[str, Any]] = None) -> str:
        """Compute cache key from query parameters."""
        key_parts = [query.lower().strip(), str(k)]
        if filters:
            key_parts.append(str(sorted(filters.items())))
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(
        self, 
        query: str, 
        k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[float, str, Dict[str, Any]]]]:
        """Get cached results if valid."""
        query_hash = self._compute_hash(query, k, filters)
        
        with self._lock:
            if query_hash not in self._cache:
                self._metrics.cache_misses += 1
                return None
            
            cached = self._cache[query_hash]
            age = time.time() - cached.timestamp
            
            if age > self._ttl_seconds:
                # Expired - remove from cache
                del self._cache[query_hash]
                self._metrics.cache_misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(query_hash)
            cached.hit_count += 1
            self._metrics.cache_hits += 1
            
            return cached.results
    
    def put(
        self,
        query: str,
        k: int,
        results: List[Tuple[float, str, Dict[str, Any]]],
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cache retrieval results."""
        query_hash = self._compute_hash(query, k, filters)
        
        with self._lock:
            cached = CachedResult(
                results=results,
                timestamp=time.time(),
                query_hash=query_hash
            )
            
            self._cache[query_hash] = cached
            self._cache.move_to_end(query_hash)
            
            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
    
    def get_metrics(self) -> RetrievalMetrics:
        """Get cache performance metrics."""
        return self._metrics


class AdaptiveRetriever:
    """Adaptive retrieval with smart parameter selection."""
    
    def __init__(self):
        self._cache = RetrievalCache()
        self._metrics = RetrievalMetrics()
    
    def compute_optimal_k(self, query: str, default_k: int = 5) -> int:
        """Compute optimal k based on query characteristics.
        
        Short queries get more results, long queries fewer.
        """
        query_len = len(query.strip())
        
        if query_len <= 10:
            # Very short - likely single word/term
            return max(default_k, 10)
        elif query_len <= 30:
            # Short phrase
            return max(default_k, 7)
        elif query_len <= 100:
            # Normal query
            return default_k
        else:
            # Long query - more specific, need fewer results
            return max(3, default_k // 2)
    
    def compute_score_threshold(self, query: str, default_threshold: float = 0.25) -> float:
        """Compute adaptive score threshold based on query."""
        query_len = len(query.strip())
        
        if query_len <= 10:
            # Short query - lower threshold
            return default_threshold * 0.7
        elif query_len <= 30:
            return default_threshold * 0.85
        else:
            return default_threshold
    
    async def retrieve_with_cache(
        self,
        query: str,
        retrieval_fn: Any,  # Async callable
        k: Optional[int] = None,
        use_cache: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Retrieve with caching and adaptive parameters.
        
        Args:
            query: Search query
            retrieval_fn: Async function that performs actual retrieval
            k: Number of results (auto-computed if None)
            use_cache: Whether to use cache
            filters: Optional filters to apply
        
        Returns:
            List of (score, content, metadata) tuples
        """
        t0 = time.perf_counter()
        
        # Compute optimal k if not provided
        if k is None:
            k = self.compute_optimal_k(query)
        
        # Try cache first
        if use_cache:
            cached = self._cache.get(query, k, filters)
            if cached is not None:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._metrics.update_latency(latency_ms)
                return cached
        
        # Cache miss - perform retrieval
        try:
            results = await retrieval_fn(query, k)
            
            # Update metrics
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._metrics.update_latency(latency_ms)
            self._metrics.total_items_retrieved += len(results)
            
            # Cache results
            if use_cache:
                self._cache.put(query, k, results, filters)
            
            return results
        except Exception:
            # On error, return empty results
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics."""
        cache_metrics = self._cache.get_metrics()
        return {
            "queries_count": self._metrics.queries_count,
            "avg_latency_ms": self._metrics.avg_latency_ms,
            "total_items": self._metrics.total_items_retrieved,
            "cache_hits": cache_metrics.cache_hits,
            "cache_misses": cache_metrics.cache_misses,
            "cache_hit_rate": (
                cache_metrics.cache_hits / max(1, cache_metrics.cache_hits + cache_metrics.cache_misses)
            ),
        }
    
    def clear_cache(self) -> None:
        """Clear retrieval cache."""
        self._cache.clear()


# Global adaptive retriever instance
_global_retriever: Optional[AdaptiveRetriever] = None
_retriever_lock = threading.RLock()


def get_adaptive_retriever() -> AdaptiveRetriever:
    """Get global adaptive retriever instance."""
    global _global_retriever
    
    if _global_retriever is not None:
        return _global_retriever
    
    with _retriever_lock:
        if _global_retriever is None:
            _global_retriever = AdaptiveRetriever()
        return _global_retriever


async def parallel_retrieve(
    query: str,
    sources: List[Any],  # List of async retrieval functions
    k: int = 5,
    max_time_ms: Optional[float] = None
) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Retrieve from multiple sources in parallel and merge results.
    
    Args:
        query: Search query
        sources: List of async functions that perform retrieval
        k: Number of results per source
        max_time_ms: Maximum time to wait for all sources
    
    Returns:
        Merged and deduplicated results sorted by score
    """
    # Launch all retrievals in parallel
    tasks = [asyncio.create_task(source(query, k)) for source in sources]
    
    # Wait with optional timeout
    if max_time_ms:
        done, pending = await asyncio.wait(
            tasks,
            timeout=max_time_ms / 1000.0,
            return_when=asyncio.ALL_COMPLETED
        )
        # Cancel pending tasks
        for task in pending:
            task.cancel()
    else:
        done = await asyncio.gather(*tasks, return_exceptions=True)
        done = [t for t in done if not isinstance(t, Exception)]
    
    # Merge results
    all_results: List[Tuple[float, str, Dict[str, Any]]] = []
    seen_content = set()
    
    for task_or_result in done:
        try:
            if isinstance(task_or_result, asyncio.Task):
                results = task_or_result.result()
            else:
                results = task_or_result
            
            for score, content, meta in results:
                # Deduplicate by content hash
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_results.append((score, content, meta))
        except Exception:
            continue
    
    # Sort by score (descending) and return top k
    all_results.sort(key=lambda x: x[0], reverse=True)
    return all_results[:k * len(sources)]  # Return k per source


__all__ = [
    'RetrievalMetrics',
    'RetrievalCache',
    'AdaptiveRetriever',
    'get_adaptive_retriever',
    'parallel_retrieve',
]
