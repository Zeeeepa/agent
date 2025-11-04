"""Intelligent semantic routing with ML-based intent detection and path optimization.

Replaces primitive string matching with semantic understanding.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from jinx.micro.embeddings.embed_cache import embed_text_cached as _embed
except Exception:
    _embed = None  # type: ignore


@dataclass
class Route:
    """Semantic route definition."""
    name: str
    patterns: List[str]  # Example patterns for learning
    handler: str  # Handler function name
    embedding_centroid: Optional[List[float]] = None
    confidence_threshold: float = 0.7
    uses: int = 0
    successes: int = 0


@dataclass
class RouteMatch:
    """Route matching result."""
    route_name: str
    confidence: float
    alternative_routes: List[Tuple[str, float]]


class SemanticRouter:
    """ML-driven semantic routing without hardcoded rules."""
    
    def __init__(self, state_path: str = "log/semantic_router.json"):
        self.state_path = state_path
        
        # Registered routes
        self.routes: Dict[str, Route] = {}
        
        # Routing history for learning
        self.history: deque[Tuple[str, str, bool]] = deque(maxlen=500)  # (query, route, success)
        
        # Dynamic route weights (learned from usage)
        self.route_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        self._lock = asyncio.Lock()
        self._load_state()
        self._register_default_routes()
    
    def _register_default_routes(self) -> None:
        """Register default semantic routes."""
        # Code execution routes
        self.routes['code_exec'] = Route(
            name='code_exec',
            patterns=[
                'run this code',
                'execute the following',
                'implement function',
                'create class',
                'write code for',
            ],
            handler='handle_code_execution',
            confidence_threshold=0.65
        )
        
        # File operations
        self.routes['file_ops'] = Route(
            name='file_ops',
            patterns=[
                'read file',
                'write to file',
                'create file',
                'delete file',
                'modify file',
            ],
            handler='handle_file_operations',
            confidence_threshold=0.70
        )
        
        # Refactoring
        self.routes['refactor'] = Route(
            name='refactor',
            patterns=[
                'refactor this code',
                'improve the structure',
                'clean up',
                'optimize function',
                'rename variable',
            ],
            handler='handle_refactoring',
            confidence_threshold=0.68
        )
        
        # Debugging
        self.routes['debug'] = Route(
            name='debug',
            patterns=[
                'fix this error',
                'debug the issue',
                'find the bug',
                'why is this failing',
                'exception occurred',
            ],
            handler='handle_debugging',
            confidence_threshold=0.72
        )
        
        # Search/Query
        self.routes['search'] = Route(
            name='search',
            patterns=[
                'find where',
                'search for',
                'locate function',
                'show me examples',
                'list all',
            ],
            handler='handle_search',
            confidence_threshold=0.65
        )
        
        # Explanation
        self.routes['explain'] = Route(
            name='explain',
            patterns=[
                'explain how',
                'what does this do',
                'how does this work',
                'describe the',
                'tell me about',
            ],
            handler='handle_explanation',
            confidence_threshold=0.60
        )
        
        # Test generation
        self.routes['test'] = Route(
            name='test',
            patterns=[
                'write tests for',
                'create unit tests',
                'add test cases',
                'test this function',
                'generate tests',
            ],
            handler='handle_test_generation',
            confidence_threshold=0.75
        )
    
    def _load_state(self) -> None:
        """Load persisted routing state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore route weights
                self.route_weights = defaultdict(lambda: 1.0, data.get('weights', {}))
                
                # Restore route stats
                for route_name, route_data in data.get('routes', {}).items():
                    if route_name in self.routes:
                        self.routes[route_name].uses = route_data.get('uses', 0)
                        self.routes[route_name].successes = route_data.get('successes', 0)
                        
                        # Restore embedding centroid
                        centroid = route_data.get('centroid')
                        if centroid:
                            self.routes[route_name].embedding_centroid = centroid
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist routing state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                routes_data = {}
                for name, route in self.routes.items():
                    routes_data[name] = {
                        'uses': route.uses,
                        'successes': route.successes,
                        'centroid': route.embedding_centroid
                    }
                
                data = {
                    'weights': dict(self.route_weights),
                    'routes': routes_data,
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def _ensure_route_embeddings(self) -> None:
        """Ensure all routes have embedding centroids."""
        if not _embed:
            return
        
        for route in self.routes.values():
            if route.embedding_centroid is None and route.patterns:
                try:
                    # Compute centroid from patterns
                    embeddings = []
                    for pattern in route.patterns[:5]:  # Limit to prevent slowdown
                        emb = await _embed(pattern)
                        if emb:
                            embeddings.append(emb)
                    
                    if embeddings:
                        # Average embeddings to get centroid
                        dim = len(embeddings[0])
                        centroid = [
                            sum(emb[i] for emb in embeddings) / len(embeddings)
                            for i in range(dim)
                        ]
                        route.embedding_centroid = centroid
                except Exception:
                    pass
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        try:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0
    
    async def route(self, query: str) -> RouteMatch:
        """Route query to appropriate handler using semantic matching."""
        async with self._lock:
            # Ensure embeddings are computed
            await self._ensure_route_embeddings()
            
            q = query.strip()
            if not q:
                return RouteMatch('explain', 0.3, [])
            
            # Get query embedding
            query_embedding: Optional[List[float]] = None
            if _embed:
                try:
                    query_embedding = await _embed(q[:512])
                except Exception:
                    pass
            
            # Score all routes
            route_scores: Dict[str, float] = {}
            
            # Semantic scoring via embeddings
            if query_embedding:
                for route_name, route in self.routes.items():
                    if route.embedding_centroid:
                        similarity = self._cosine_similarity(query_embedding, route.embedding_centroid)
                        
                        # Apply learned weights
                        weight = self.route_weights[route_name]
                        
                        # Success rate boost
                        if route.uses > 0:
                            success_rate = route.successes / route.uses
                            similarity *= (1.0 + success_rate * 0.3)
                        
                        route_scores[route_name] = similarity * weight
            
            # Fallback: keyword-based scoring
            if not route_scores:
                q_lower = q.lower()
                for route_name, route in self.routes.items():
                    score = 0.0
                    for pattern in route.patterns:
                        if pattern.lower() in q_lower:
                            score += 1.0
                        # Partial match
                        pattern_words = pattern.lower().split()
                        matches = sum(1 for w in pattern_words if w in q_lower)
                        if matches > 0:
                            score += matches / len(pattern_words) * 0.5
                    
                    if score > 0:
                        route_scores[route_name] = score * self.route_weights[route_name]
            
            # Select best route
            if not route_scores:
                return RouteMatch('explain', 0.3, [])
            
            sorted_routes = sorted(route_scores.items(), key=lambda x: x[1], reverse=True)
            best_route, best_score = sorted_routes[0]
            
            # Normalize scores
            max_score = max(route_scores.values())
            if max_score > 0:
                confidence = min(1.0, best_score / max_score)
            else:
                confidence = 0.3
            
            # Alternative routes
            alternatives = [
                (name, min(1.0, score / max_score))
                for name, score in sorted_routes[1:4]
                if score > 0.3 * max_score
            ]
            
            return RouteMatch(best_route, confidence, alternatives)
    
    async def record_outcome(self, query: str, route_name: str, success: bool) -> None:
        """Record routing outcome for learning."""
        async with self._lock:
            self.history.append((query, route_name, success))
            
            if route_name in self.routes:
                route = self.routes[route_name]
                route.uses += 1
                if success:
                    route.successes += 1
                
                # Update weight using EMA
                alpha = 0.1
                new_weight = 1.5 if success else 0.7
                current_weight = self.route_weights[route_name]
                self.route_weights[route_name] = alpha * new_weight + (1 - alpha) * current_weight
            
            # Periodically save
            if len(self.history) % 20 == 0:
                await self._save_state()
    
    def get_stats(self) -> Dict[str, object]:
        """Get routing statistics."""
        stats = {}
        for name, route in self.routes.items():
            success_rate = route.successes / route.uses if route.uses > 0 else 0.0
            stats[name] = {
                'uses': route.uses,
                'successes': route.successes,
                'success_rate': success_rate,
                'weight': self.route_weights[name]
            }
        
        return {
            'routes': stats,
            'total_routings': len(self.history)
        }


# Singleton
_router: Optional[SemanticRouter] = None
_router_lock = asyncio.Lock()


async def get_semantic_router() -> SemanticRouter:
    """Get singleton semantic router."""
    global _router
    if _router is None:
        async with _router_lock:
            if _router is None:
                _router = SemanticRouter()
    return _router


async def route_query(query: str) -> RouteMatch:
    """Route query semantically."""
    router = await get_semantic_router()
    return await router.route(query)


async def record_routing_outcome(query: str, route_name: str, success: bool) -> None:
    """Record routing outcome."""
    router = await get_semantic_router()
    await router.record_outcome(query, route_name, success)


__all__ = [
    "SemanticRouter",
    "RouteMatch",
    "get_semantic_router",
    "route_query",
    "record_routing_outcome",
]
