"""Intelligent query classification using embeddings and pattern learning.

Replaces primitive if/else chains with ML-driven intent classification.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    from jinx.micro.embeddings.embed_cache import embed_text_cached as _embed
except Exception:
    _embed = None  # type: ignore


@dataclass
class QueryIntent:
    """Classified query intent with confidence."""
    intent: str  # 'code_exec', 'refactor', 'explain', 'debug', 'search', 'chat'
    confidence: float
    sub_intents: List[Tuple[str, float]]  # secondary intents
    features: Dict[str, float]


class IntelligentQueryClassifier:
    """ML-based query intent classifier without hardcoded rules."""
    
    def __init__(self, state_path: str = "log/query_classifier.json"):
        self.state_path = state_path
        
        # Intent-specific semantic centroids (learned from outcomes)
        self.intent_centroids: Dict[str, List[float]] = {}
        
        # Intent occurrence counts for prior probability
        self.intent_counts: Dict[str, int] = defaultdict(int)
        
        # Co-occurrence matrix for sub-intent detection
        self.intent_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Structural features learned per intent
        self.intent_features: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted classifier state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.intent_centroids = data.get('centroids', {})
                self.intent_counts = defaultdict(int, data.get('counts', {}))
                
                # Restore co-occurrence
                cooc_data = data.get('cooccurrence', {})
                for key_str, count in cooc_data.items():
                    i1, i2 = key_str.split('|', 1)
                    self.intent_cooccurrence[(i1, i2)] = count
                
                # Restore features
                feat_data = data.get('features', {})
                for intent, features in feat_data.items():
                    self.intent_features[intent] = defaultdict(float, features)
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist classifier state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize co-occurrence
                cooc_data = {f"{k1}|{k2}": v for (k1, k2), v in self.intent_cooccurrence.items()}
                
                data = {
                    'centroids': self.intent_centroids,
                    'counts': dict(self.intent_counts),
                    'cooccurrence': cooc_data,
                    'features': {k: dict(v) for k, v in self.intent_features.items()},
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _extract_structural_features(self, query: str) -> Dict[str, float]:
        """Extract language-agnostic structural features."""
        features: Dict[str, float] = {}
        
        q = query.strip()
        if not q:
            return features
        
        # Length features
        features['len_chars'] = min(1.0, len(q) / 500.0)
        features['len_words'] = min(1.0, len(q.split()) / 50.0)
        
        # Punctuation density
        punct = sum(1 for c in q if c in '?!.,;:')
        features['punct_density'] = min(1.0, punct / max(1, len(q)) * 10)
        
        # Question markers
        features['has_question'] = 1.0 if '?' in q else 0.0
        
        # Code indicators (structural, not keywords)
        code_chars = sum(1 for c in q if c in '(){}[]<>=+-*/|&%^~')
        features['code_density'] = min(1.0, code_chars / max(1, len(q)) * 5)
        
        # Line count (multi-line suggests code/data)
        lines = q.split('\n')
        features['multi_line'] = min(1.0, len(lines) / 10.0)
        
        # Indentation presence
        indented = sum(1 for ln in lines if ln.startswith((' ', '\t')))
        features['has_indent'] = min(1.0, indented / max(1, len(lines)))
        
        # Uppercase ratio (acronyms, constants)
        alpha = [c for c in q if c.isalpha()]
        if alpha:
            features['upper_ratio'] = sum(1 for c in alpha if c.isupper()) / len(alpha)
        
        # Number presence
        features['has_numbers'] = 1.0 if any(c.isdigit() for c in q) else 0.0
        
        return features
    
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
    
    async def classify(self, query: str) -> QueryIntent:
        """Classify query intent using learned patterns."""
        async with self._lock:
            q = query.strip()
            if not q:
                return QueryIntent('chat', 0.5, [], {})
            
            # Extract structural features
            features = self._extract_structural_features(q)
            
            # Get query embedding if available
            embedding: Optional[List[float]] = None
            if _embed:
                try:
                    embedding = await _embed(q[:512])
                except Exception:
                    pass
            
            # Compute intent scores
            intent_scores: Dict[str, float] = {}
            
            # If we have embeddings and learned centroids
            if embedding and self.intent_centroids:
                for intent, centroid in self.intent_centroids.items():
                    sim = self._cosine_similarity(embedding, centroid)
                    intent_scores[intent] = sim
            
            # Add feature-based scoring
            for intent, learned_features in self.intent_features.items():
                feature_score = 0.0
                for feat_name, feat_val in features.items():
                    if feat_name in learned_features:
                        # How well does this feature match learned pattern?
                        expected = learned_features[feat_name]
                        diff = abs(feat_val - expected)
                        feature_score += (1.0 - diff) * 0.1
                
                intent_scores[intent] = intent_scores.get(intent, 0.0) + feature_score
            
            # Apply prior probability (more frequent intents get small boost)
            total_counts = sum(self.intent_counts.values())
            if total_counts > 0:
                for intent, count in self.intent_counts.items():
                    prior = count / total_counts
                    intent_scores[intent] = intent_scores.get(intent, 0.0) + prior * 0.2
            
            # Default fallback intents if nothing learned yet
            if not intent_scores:
                # Use structural features for cold start
                if features.get('code_density', 0) > 0.3:
                    intent_scores['code_exec'] = 0.6
                elif features.get('has_question', 0) > 0.5:
                    intent_scores['explain'] = 0.5
                else:
                    intent_scores['chat'] = 0.5
            
            # Normalize scores
            max_score = max(intent_scores.values()) if intent_scores else 1.0
            if max_score > 0:
                intent_scores = {k: v / max_score for k, v in intent_scores.items()}
            
            # Primary intent
            if intent_scores:
                primary_intent = max(intent_scores.items(), key=lambda x: x[1])
                primary_name, primary_conf = primary_intent
            else:
                primary_name, primary_conf = 'chat', 0.5
            
            # Sub-intents (above threshold)
            sub_intents = [
                (intent, score) 
                for intent, score in sorted(intent_scores.items(), key=lambda x: -x[1])
                if intent != primary_name and score > 0.3
            ][:3]  # Top 3 sub-intents
            
            return QueryIntent(
                intent=primary_name,
                confidence=primary_conf,
                sub_intents=sub_intents,
                features=features
            )
    
    async def learn(self, query: str, actual_intent: str, outcome_quality: float = 1.0) -> None:
        """Learn from classified query and outcome."""
        async with self._lock:
            q = query.strip()
            if not q:
                return
            
            # Update counts
            self.intent_counts[actual_intent] += 1
            
            # Extract and store features
            features = self._extract_structural_features(q)
            
            # Update feature expectations using exponential moving average
            alpha = 0.1
            for feat_name, feat_val in features.items():
                current = self.intent_features[actual_intent][feat_name]
                self.intent_features[actual_intent][feat_name] = alpha * feat_val + (1 - alpha) * current
            
            # Update centroid if we have embeddings
            if _embed:
                try:
                    embedding = await _embed(q[:512])
                    if embedding:
                        if actual_intent in self.intent_centroids:
                            # Update centroid using moving average
                            old_centroid = self.intent_centroids[actual_intent]
                            new_centroid = [
                                alpha * e + (1 - alpha) * old
                                for e, old in zip(embedding, old_centroid)
                            ]
                            self.intent_centroids[actual_intent] = new_centroid
                        else:
                            # Initialize centroid
                            self.intent_centroids[actual_intent] = embedding
                except Exception:
                    pass
            
            # Periodically save
            if sum(self.intent_counts.values()) % 10 == 0:
                await self._save_state()
    
    def get_stats(self) -> Dict[str, object]:
        """Get classifier statistics."""
        total = sum(self.intent_counts.values())
        
        return {
            'total_queries': total,
            'intent_distribution': {
                k: {'count': v, 'ratio': v / total if total > 0 else 0.0}
                for k, v in sorted(self.intent_counts.items(), key=lambda x: -x[1])
            },
            'learned_centroids': len(self.intent_centroids),
            'learned_features': sum(len(v) for v in self.intent_features.values())
        }


# Singleton
_classifier: Optional[IntelligentQueryClassifier] = None
_classifier_lock = asyncio.Lock()


async def get_query_classifier() -> IntelligentQueryClassifier:
    """Get singleton query classifier."""
    global _classifier
    if _classifier is None:
        async with _classifier_lock:
            if _classifier is None:
                _classifier = IntelligentQueryClassifier()
    return _classifier


async def classify_query(query: str) -> QueryIntent:
    """Classify query intent."""
    classifier = await get_query_classifier()
    return await classifier.classify(query)


async def learn_query_intent(query: str, intent: str, quality: float = 1.0) -> None:
    """Learn from query outcome."""
    classifier = await get_query_classifier()
    await classifier.learn(query, intent, quality)


__all__ = [
    "QueryIntent",
    "IntelligentQueryClassifier",
    "get_query_classifier",
    "classify_query",
    "learn_query_intent",
]
