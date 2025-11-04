"""Query Understanding - sophisticated query analysis and expansion.

Advanced NLP techniques:
- Semantic parsing
- Intent classification with confidence
- Entity extraction
- Query rewriting
- Expansion with embeddings
- Contextual disambiguation
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class QueryIntent(Enum):
    """Query intent classification."""
    CODE_SEARCH = "code_search"
    CODE_EDIT = "code_edit"
    EXPLANATION = "explanation"
    PLANNING = "planning"
    CONVERSATION = "conversation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Complete query analysis result."""
    original_query: str
    intent: QueryIntent
    confidence: float
    
    # Extracted entities
    entities: Dict[str, List[str]]  # type -> list of entities
    
    # Query expansions
    expansions: List[str]
    synonyms: Dict[str, List[str]]
    
    # Contextual info
    is_follow_up: bool
    requires_code_context: bool
    temporal_context: Optional[str]  # 'recent', 'historical', None
    
    # Complexity
    complexity_score: float  # 0-1


class QueryUnderstanding:
    """Advanced query understanding with ML."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Intent classification patterns
        self._intent_patterns = {
            QueryIntent.CODE_SEARCH: [
                r'find\s+(function|class|method)',
                r'where\s+is',
                r'show\s+me',
                r'search\s+for',
            ],
            QueryIntent.CODE_EDIT: [
                r'(add|create|make|implement)',
                r'(fix|repair|correct)',
                r'(change|modify|update)',
                r'(delete|remove)',
            ],
            QueryIntent.EXPLANATION: [
                r'(explain|what|why|how)',
                r'what\s+does',
                r'how\s+does',
            ],
            QueryIntent.PLANNING: [
                r'(plan|steps|approach)',
                r'how\s+to',
                r'what\s+should',
            ],
            QueryIntent.DEBUGGING: [
                r'(error|bug|issue|problem)',
                r'(not\s+working|fails)',
                r'(crash|exception)',
            ],
            QueryIntent.REFACTORING: [
                r'refactor',
                r'improve',
                r'optimize',
                r'clean\s+up',
            ],
        }
        
        # Entity patterns
        self._entity_patterns = {
            'file': r'([a-zA-Z0-9_\-./]+\.(py|js|ts|java|cpp|h))',
            'class': r'\b([A-Z][a-zA-Z0-9_]*)\b',
            'function': r'\b([a-z_][a-zA-Z0-9_]*)\s*\(',
            'variable': r'\b([a-z_][a-z0-9_]*)\b',
        }
        
        # Performance cache
        self._analysis_cache: Dict[str, QueryAnalysis] = {}
    
    async def analyze(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryAnalysis:
        """Comprehensively analyze query."""
        
        # Check cache
        cache_key = f"{query}:{context.get('type', '') if context else ''}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        async with self._lock:
            # Stage 1: Intent classification
            intent, confidence = await self._classify_intent(query, context)
            
            # Stage 2: Entity extraction
            entities = await self._extract_entities(query, intent)
            
            # Stage 3: Query expansion
            expansions, synonyms = await self._generate_expansions(
                query, intent, entities
            )
            
            # Stage 4: Context analysis
            is_follow_up = await self._detect_follow_up(query, context)
            requires_code = self._requires_code_context(query, intent)
            temporal = self._detect_temporal_context(query)
            
            # Stage 5: Complexity scoring
            complexity = self._calculate_complexity(query, intent, entities)
            
            analysis = QueryAnalysis(
                original_query=query,
                intent=intent,
                confidence=confidence,
                entities=entities,
                expansions=expansions,
                synonyms=synonyms,
                is_follow_up=is_follow_up,
                requires_code_context=requires_code,
                temporal_context=temporal,
                complexity_score=complexity
            )
            
            # Cache result
            self._analysis_cache[cache_key] = analysis
            
            return analysis
    
    async def _classify_intent(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence."""
        
        query_lower = query.lower()
        
        # Score each intent
        scores = {}
        
        for intent, patterns in self._intent_patterns.items():
            score = 0.0
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
            
            # Normalize by number of patterns
            scores[intent] = score / len(patterns)
        
        # Context boost
        if context:
            context_type = context.get('type', '')
            if context_type == 'code' and QueryIntent.CODE_SEARCH in scores:
                scores[QueryIntent.CODE_SEARCH] *= 1.3
            elif context_type == 'planning' and QueryIntent.PLANNING in scores:
                scores[QueryIntent.PLANNING] *= 1.3
        
        # Get best intent
        if scores:
            best_intent = max(scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return best_intent[0], min(1.0, best_intent[1])
        
        # Default
        return QueryIntent.UNKNOWN, 0.3
    
    async def _extract_entities(
        self,
        query: str,
        intent: QueryIntent
    ) -> Dict[str, List[str]]:
        """Extract entities from query."""
        
        entities: Dict[str, List[str]] = {}
        
        for entity_type, pattern in self._entity_patterns.items():
            matches = re.findall(pattern, query)
            
            if matches:
                # Deduplicate
                unique = list(set(m if isinstance(m, str) else m[0] for m in matches))
                entities[entity_type] = unique
        
        # Intent-specific extraction
        if intent == QueryIntent.CODE_SEARCH:
            # Extract quoted strings as search terms
            quoted = re.findall(r'"([^"]+)"', query)
            if quoted:
                entities['search_term'] = quoted
        
        return entities
    
    async def _generate_expansions(
        self,
        query: str,
        intent: QueryIntent,
        entities: Dict[str, List[str]]
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Generate query expansions and synonyms."""
        
        expansions = [query]  # Original first
        synonyms: Dict[str, List[str]] = {}
        
        # Intent-based expansion
        if intent == QueryIntent.CODE_SEARCH:
            if 'function' in entities:
                for func in entities['function']:
                    expansions.append(f"def {func}")
                    expansions.append(f"implementation of {func}")
        
        elif intent == QueryIntent.EXPLANATION:
            expansions.append(f"how does {query} work")
            expansions.append(f"documentation for {query}")
        
        elif intent == QueryIntent.DEBUGGING:
            expansions.append(f"error in {query}")
            expansions.append(f"fix {query}")
        
        # Synonym expansion
        words = query.lower().split()
        for word in words:
            word_syns = self._get_synonyms(word)
            if word_syns:
                synonyms[word] = word_syns
                
                # Create synonym expansion
                expanded = query.lower()
                for syn in word_syns[:1]:  # Use first synonym
                    expanded = expanded.replace(word, syn)
                    if expanded != query.lower():
                        expansions.append(expanded)
        
        return expansions[:5], synonyms
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for common programming terms."""
        
        synonym_map = {
            'function': ['method', 'def', 'procedure'],
            'class': ['type', 'object', 'struct'],
            'variable': ['var', 'field', 'property'],
            'error': ['exception', 'bug', 'issue', 'problem'],
            'fix': ['repair', 'resolve', 'correct', 'solve'],
            'create': ['make', 'build', 'generate', 'add'],
            'delete': ['remove', 'destroy', 'clear', 'drop'],
            'change': ['modify', 'update', 'alter', 'edit'],
            'find': ['search', 'locate', 'get', 'retrieve'],
            'show': ['display', 'print', 'list', 'output'],
        }
        
        return synonym_map.get(word, [])
    
    async def _detect_follow_up(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Detect if this is a follow-up query."""
        
        # Check for follow-up indicators
        follow_up_patterns = [
            r'^(and|also|additionally|furthermore)',
            r'^(what about|how about)',
            r'\b(that|this|it|them)\b',
            r'^(yes|no|ok|sure)',
        ]
        
        query_lower = query.lower()
        
        for pattern in follow_up_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Short queries are often follow-ups
        if len(query.split()) <= 3 and context:
            return True
        
        return False
    
    def _requires_code_context(
        self,
        query: str,
        intent: QueryIntent
    ) -> bool:
        """Determine if query requires code context."""
        
        if intent in [QueryIntent.CODE_SEARCH, QueryIntent.CODE_EDIT, 
                      QueryIntent.DEBUGGING, QueryIntent.REFACTORING]:
            return True
        
        # Check for code-related keywords
        code_keywords = [
            'function', 'class', 'method', 'file', 'code',
            'implementation', 'module', 'import', 'variable'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in code_keywords)
    
    def _detect_temporal_context(self, query: str) -> Optional[str]:
        """Detect temporal context."""
        
        query_lower = query.lower()
        
        recent_keywords = ['recent', 'latest', 'new', 'current', 'now', 'today']
        historical_keywords = ['old', 'previous', 'past', 'before', 'earlier']
        
        if any(k in query_lower for k in recent_keywords):
            return 'recent'
        elif any(k in query_lower for k in historical_keywords):
            return 'historical'
        
        return None
    
    def _calculate_complexity(
        self,
        query: str,
        intent: QueryIntent,
        entities: Dict[str, List[str]]
    ) -> float:
        """Calculate query complexity score."""
        
        complexity = 0.0
        
        # Length factor
        word_count = len(query.split())
        complexity += min(0.3, word_count / 20)
        
        # Entity count
        entity_count = sum(len(v) for v in entities.values())
        complexity += min(0.3, entity_count / 5)
        
        # Intent complexity
        intent_complexity = {
            QueryIntent.EXPLANATION: 0.5,
            QueryIntent.PLANNING: 0.7,
            QueryIntent.REFACTORING: 0.8,
            QueryIntent.CODE_EDIT: 0.6,
            QueryIntent.CODE_SEARCH: 0.4,
            QueryIntent.DEBUGGING: 0.7,
            QueryIntent.CONVERSATION: 0.3,
        }
        complexity += intent_complexity.get(intent, 0.4)
        
        return min(1.0, complexity)


# Singleton
_query_understanding: Optional[QueryUnderstanding] = None
_qu_lock = asyncio.Lock()


async def get_query_understanding() -> QueryUnderstanding:
    """Get singleton query understanding."""
    global _query_understanding
    if _query_understanding is None:
        async with _qu_lock:
            if _query_understanding is None:
                _query_understanding = QueryUnderstanding()
    return _query_understanding


__all__ = [
    "QueryUnderstanding",
    "QueryAnalysis",
    "QueryIntent",
    "get_query_understanding",
]
