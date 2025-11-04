"""Query Expander - расширяет запросы для лучших результатов.

Использует ML для автоматического enrichment запросов.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ExpandedQuery:
    """Expanded query with metadata."""
    original: str
    expanded: str
    expansions: List[str]
    confidence: float
    method: str  # 'synonym', 'context', 'semantic', 'hybrid'


class QueryExpander:
    """ML-driven query expansion for better retrieval."""
    
    def __init__(self, state_path: str = "log/query_expander.json"):
        self.state_path = state_path
        
        # Learned expansions
        self.expansion_patterns: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        # Synonym dictionary (learned)
        self.synonyms: Dict[str, Set[str]] = defaultdict(set)
        
        # Co-occurrence matrix
        self.cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Expansion history
        self.history: deque[Tuple[str, str, bool]] = deque(maxlen=500)
        
        # Success rates per method
        self.method_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'uses': 0,
            'successes': 0
        })
        
        self._lock = asyncio.Lock()
        self._load_state()
        self._initialize_base_synonyms()
    
    def _initialize_base_synonyms(self) -> None:
        """Initialize base programming synonyms."""
        base_synonyms = {
            'function': {'func', 'method', 'def', 'procedure'},
            'class': {'type', 'object', 'struct'},
            'variable': {'var', 'val', 'const'},
            'error': {'exception', 'bug', 'issue', 'problem'},
            'fix': {'repair', 'solve', 'correct', 'patch'},
            'create': {'make', 'build', 'generate', 'construct'},
            'delete': {'remove', 'drop', 'erase'},
            'update': {'modify', 'change', 'edit', 'alter'},
            'find': {'search', 'locate', 'discover', 'get'},
            'async': {'asynchronous', 'concurrent', 'parallel'},
        }
        
        for word, syns in base_synonyms.items():
            self.synonyms[word].update(syns)
            for syn in syns:
                self.synonyms[syn].add(word)
    
    def _load_state(self) -> None:
        """Load expander state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore expansion patterns
                for term, patterns in data.get('patterns', {}).items():
                    self.expansion_patterns[term] = [tuple(p) for p in patterns]
                
                # Restore synonyms
                for word, syns in data.get('synonyms', {}).items():
                    self.synonyms[word] = set(syns)
                
                # Restore method performance
                for method, perf in data.get('methods', {}).items():
                    self.method_performance[method] = perf
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist expander state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize patterns
                patterns_data = {
                    term: list(patterns)
                    for term, patterns in self.expansion_patterns.items()
                }
                
                # Serialize synonyms
                synonyms_data = {
                    word: list(syns)
                    for word, syns in self.synonyms.items()
                }
                
                data = {
                    'patterns': patterns_data,
                    'synonyms': synonyms_data,
                    'methods': dict(self.method_performance),
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _extract_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        terms = cleaned.split()
        
        # Filter stop words and short terms
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'this', 'that'}
        terms = [t for t in terms if t not in stop_words and len(t) > 2]
        
        return terms
    
    async def expand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExpandedQuery:
        """Expand query using multiple strategies."""
        async with self._lock:
            q = query.strip()
            if not q:
                return ExpandedQuery(q, q, [], 0.0, 'none')
            
            # Extract terms
            terms = self._extract_terms(q)
            
            # Try multiple expansion methods
            synonym_exp = await self._expand_synonyms(terms)
            pattern_exp = await self._expand_patterns(terms)
            context_exp = await self._expand_context(terms, context)
            
            # Select best method
            method = await self._select_method(query, context)
            
            # Combine expansions based on method
            if method == 'synonym':
                expansions = synonym_exp
            elif method == 'pattern':
                expansions = pattern_exp
            elif method == 'context':
                expansions = context_exp
            else:  # hybrid
                expansions = list(set(synonym_exp + pattern_exp + context_exp))
            
            # Build expanded query
            expanded_terms = list(set([q] + expansions[:5]))  # Original + top 5
            expanded = ' '.join(expanded_terms)
            
            # Compute confidence
            confidence = min(1.0, len(expansions) / 10.0)
            
            return ExpandedQuery(
                original=q,
                expanded=expanded,
                expansions=expansions,
                confidence=confidence,
                method=method
            )
    
    async def _expand_synonyms(self, terms: List[str]) -> List[str]:
        """Expand using synonyms."""
        expansions = []
        
        for term in terms:
            # Get synonyms
            syns = self.synonyms.get(term, set())
            expansions.extend(list(syns)[:3])  # Top 3 per term
        
        return expansions
    
    async def _expand_patterns(self, terms: List[str]) -> List[str]:
        """Expand using learned patterns."""
        expansions = []
        
        for term in terms:
            # Get learned patterns
            patterns = self.expansion_patterns.get(term, [])
            
            # Sort by weight
            patterns.sort(key=lambda x: x[1], reverse=True)
            
            # Take top patterns
            expansions.extend([p[0] for p in patterns[:3]])
        
        return expansions
    
    async def _expand_context(
        self,
        terms: List[str],
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Expand using context."""
        expansions = []
        
        if not context:
            return expansions
        
        # Use intent to add relevant terms
        intent = context.get('intent', '')
        
        if intent == 'code_exec':
            expansions.extend(['function', 'class', 'method', 'implement'])
        elif intent == 'debug':
            expansions.extend(['error', 'exception', 'fix', 'bug'])
        elif intent == 'refactor':
            expansions.extend(['optimize', 'improve', 'clean', 'restructure'])
        elif intent == 'search':
            expansions.extend(['find', 'locate', 'where', 'usage'])
        
        return expansions
    
    async def _select_method(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Select best expansion method."""
        # Compute scores for each method
        scores = {}
        
        for method, perf in self.method_performance.items():
            if perf['uses'] > 0:
                success_rate = perf['successes'] / perf['uses']
                scores[method] = success_rate
        
        # If no history, use heuristics
        if not scores:
            if context and context.get('intent'):
                return 'context'
            return 'hybrid'
        
        # Select best
        best_method = max(scores.items(), key=lambda x: x[1])[0]
        
        return best_method
    
    async def record_outcome(
        self,
        original_query: str,
        expanded_query: str,
        method: str,
        success: bool
    ) -> None:
        """Record expansion outcome."""
        async with self._lock:
            # Record history
            self.history.append((original_query, expanded_query, success))
            
            # Update method performance
            perf = self.method_performance[method]
            perf['uses'] += 1
            if success:
                perf['successes'] += 1
            
            # Learn patterns on success
            if success:
                orig_terms = self._extract_terms(original_query)
                exp_terms = self._extract_terms(expanded_query)
                
                # Find new terms in expansion
                new_terms = [t for t in exp_terms if t not in orig_terms]
                
                # Associate new terms with original terms
                for orig_term in orig_terms:
                    for new_term in new_terms:
                        # Add to patterns
                        patterns = self.expansion_patterns[orig_term]
                        
                        # Find existing
                        existing = next((p for p in patterns if p[0] == new_term), None)
                        
                        if existing:
                            # Increase weight
                            idx = patterns.index(existing)
                            patterns[idx] = (new_term, existing[1] + 0.1)
                        else:
                            # Add new
                            patterns.append((new_term, 0.5))
                        
                        # Keep top 10
                        patterns.sort(key=lambda x: x[1], reverse=True)
                        self.expansion_patterns[orig_term] = patterns[:10]
                        
                        # Learn as synonym if strong pattern
                        if existing and existing[1] > 0.8:
                            self.synonyms[orig_term].add(new_term)
                            self.synonyms[new_term].add(orig_term)
            
            # Periodically save
            if len(self.history) % 20 == 0:
                await self._save_state()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get expander statistics."""
        return {
            'expansions_performed': len(self.history),
            'learned_patterns': sum(len(p) for p in self.expansion_patterns.values()),
            'synonyms_learned': sum(len(s) for s in self.synonyms.values()),
            'method_performance': {
                k: {
                    'success_rate': v['successes'] / max(1, v['uses']),
                    'uses': v['uses']
                }
                for k, v in self.method_performance.items()
                if v['uses'] > 0
            }
        }


# Singleton
_expander: Optional[QueryExpander] = None
_expander_lock = asyncio.Lock()


async def get_query_expander() -> QueryExpander:
    """Get singleton query expander."""
    global _expander
    if _expander is None:
        async with _expander_lock:
            if _expander is None:
                _expander = QueryExpander()
    return _expander


async def expand_query(query: str, context: Optional[Dict[str, Any]] = None) -> ExpandedQuery:
    """Expand query for better retrieval."""
    expander = await get_query_expander()
    return await expander.expand(query, context)


async def record_expansion_outcome(original: str, expanded: str, method: str, success: bool) -> None:
    """Record expansion outcome."""
    expander = await get_query_expander()
    await expander.record_outcome(original, expanded, method, success)


__all__ = [
    "QueryExpander",
    "ExpandedQuery",
    "get_query_expander",
    "expand_query",
    "record_expansion_outcome",
]
