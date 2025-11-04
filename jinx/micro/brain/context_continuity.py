"""Context Continuity - intelligent conversation flow and context tracking.

Understands when user asks follow-up questions, references previous code, or clarifies.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    id: str
    user_query: str
    system_response: str
    intent: str
    code_files: Set[str]
    topics: List[str]
    timestamp: float


class ContextContinuity:
    """Tracks and understands conversation continuity."""
    
    def __init__(self, history_size: int = 20):
        self.history: deque[ConversationTurn] = deque(maxlen=history_size)
        
        # Current session tracking
        self.current_topic: Optional[str] = None
        self.active_code_files: Set[str] = set()
        self.user_goals: List[str] = []
        
        self._lock = asyncio.Lock()
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for continuity and context."""
        async with self._lock:
            analysis = {
                'is_new_topic': False,
                'references_previous': False,
                'is_clarification': False,
                'is_follow_up': False,
                'related_turns': [],
                'inferred_context': {},
                'suggested_context': ''
            }
            
            # Check if references previous conversation
            if await self._references_previous(query):
                analysis['references_previous'] = True
                analysis['related_turns'] = await self._find_related_turns(query)
            
            # Check if clarification
            if await self._is_clarification(query):
                analysis['is_clarification'] = True
                if self.history:
                    analysis['clarifying_turn'] = self.history[-1].id
            
            # Check if follow-up
            if await self._is_follow_up(query):
                analysis['is_follow_up'] = True
            
            # Check if new topic
            if await self._is_new_topic(query):
                analysis['is_new_topic'] = True
            
            # Infer missing context
            inferred = await self._infer_context(query, analysis)
            analysis['inferred_context'] = inferred
            
            # Build suggested context string
            if analysis['references_previous'] and analysis['related_turns']:
                prev_queries = [
                    self._get_turn(turn_id).user_query 
                    for turn_id in analysis['related_turns'][:2]
                    if self._get_turn(turn_id)
                ]
                analysis['suggested_context'] = " | Previous: " + " -> ".join(prev_queries)
            
            return analysis
    
    async def record_turn(
        self,
        user_query: str,
        system_response: str,
        intent: str,
        code_files: Optional[Set[str]] = None,
        topics: Optional[List[str]] = None
    ) -> str:
        """Record conversation turn."""
        async with self._lock:
            turn_id = f"turn_{len(self.history)}"
            
            turn = ConversationTurn(
                id=turn_id,
                user_query=user_query,
                system_response=system_response,
                intent=intent,
                code_files=code_files or set(),
                topics=topics or [],
                timestamp=time.time()
            )
            
            self.history.append(turn)
            
            # Update active tracking
            if code_files:
                self.active_code_files.update(code_files)
            
            if topics:
                self.current_topic = topics[-1] if topics else self.current_topic
            
            return turn_id
    
    async def _references_previous(self, query: str) -> bool:
        """Check if query references previous conversation."""
        q_lower = query.lower()
        
        reference_indicators = [
            'that', 'this', 'it', 'them', 'those', 'these',
            'previous', 'last', 'above', 'before', 'earlier',
            'same', 'there', 'here', 'what you', 'you said',
            'you mentioned', 'as you', 'like you'
        ]
        
        return any(indicator in q_lower for indicator in reference_indicators)
    
    async def _is_clarification(self, query: str) -> bool:
        """Check if query is clarifying previous response."""
        q_lower = query.lower()
        
        clarification_patterns = [
            'what do you mean', 'clarify', 'explain', 'confused',
            'not clear', "don't understand", 'can you elaborate',
            'more details', 'be more specific', 'wait', 'hold on'
        ]
        
        return any(pattern in q_lower for pattern in clarification_patterns)
    
    async def _is_follow_up(self, query: str) -> bool:
        """Check if query is follow-up to previous topic."""
        if not self.history:
            return False
        
        q_lower = query.lower()
        
        # Check for continuation words
        continuation_words = [
            'also', 'and', 'additionally', 'moreover',
            'furthermore', 'too', 'as well', 'plus',
            'next', 'then', 'after that'
        ]
        
        if any(word in q_lower for word in continuation_words):
            return True
        
        # Check if mentions same files/topics as recent
        recent = list(self.history)[-3:]
        recent_files = set()
        for turn in recent:
            recent_files.update(turn.code_files)
        
        # Simple file mention check
        if recent_files:
            for file in recent_files:
                if file in query:
                    return True
        
        return False
    
    async def _is_new_topic(self, query: str) -> bool:
        """Check if query introduces new topic."""
        if not self.history:
            return True
        
        # Check for topic-change indicators
        q_lower = query.lower()
        
        new_topic_indicators = [
            'now', 'different', 'instead', 'switch to',
            'change to', 'new', 'another', 'other',
            'forget that', 'never mind', 'actually'
        ]
        
        return any(indicator in q_lower for indicator in new_topic_indicators)
    
    async def _find_related_turns(self, query: str) -> List[str]:
        """Find conversation turns related to query."""
        related = []
        
        if not self.history:
            return related
        
        # Get recent turns (last 5)
        recent = list(self.history)[-5:]
        
        # Simple relevance: check for common words or file mentions
        q_words = set(query.lower().split())
        
        for turn in recent:
            turn_words = set(turn.user_query.lower().split())
            
            # Check word overlap
            overlap = len(q_words & turn_words)
            if overlap >= 2:
                related.append(turn.id)
                continue
            
            # Check file mentions
            for file in turn.code_files:
                if file in query:
                    related.append(turn.id)
                    break
        
        return related
    
    async def _infer_context(
        self,
        query: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Infer missing context from conversation history."""
        context = {}
        
        if not self.history:
            return context
        
        # If references previous but context unclear, infer from last turn
        if analysis['references_previous'] and not analysis.get('related_turns'):
            last_turn = self.history[-1]
            context['inferred_from_last'] = {
                'query': last_turn.user_query,
                'intent': last_turn.intent,
                'files': list(last_turn.code_files)
            }
        
        # If mentions "it" or "that", infer subject from recent turns
        q_lower = query.lower()
        if 'it' in q_lower or 'that' in q_lower:
            # Look for subjects in recent turns
            recent = list(self.history)[-3:]
            for turn in reversed(recent):
                if turn.code_files:
                    context['inferred_subject'] = list(turn.code_files)[0]
                    break
                if turn.topics:
                    context['inferred_subject'] = turn.topics[0]
                    break
        
        # Infer active files if working on code
        if self.active_code_files and 'code' in q_lower:
            context['active_files'] = list(self.active_code_files)
        
        return context
    
    def _get_turn(self, turn_id: str) -> Optional[ConversationTurn]:
        """Get turn by ID."""
        for turn in self.history:
            if turn.id == turn_id:
                return turn
        return None
    
    async def get_context_for_query(self, query: str) -> str:
        """Get full context string for query."""
        analysis = await self.analyze_query(query)
        
        parts = []
        
        # Add suggested context
        if analysis.get('suggested_context'):
            parts.append(analysis['suggested_context'])
        
        # Add inferred context
        inferred = analysis.get('inferred_context', {})
        if inferred.get('inferred_from_last'):
            prev = inferred['inferred_from_last']
            parts.append(f" | Context: {prev['intent']} on {', '.join(prev['files'][:2])}")
        
        if inferred.get('active_files'):
            parts.append(f" | Active files: {', '.join(inferred['active_files'][:3])}")
        
        return "".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            'turns': len(self.history),
            'current_topic': self.current_topic,
            'active_files': len(self.active_code_files),
            'recent_intents': [t.intent for t in list(self.history)[-5:]]
        }


# Singleton
_continuity: Optional[ContextContinuity] = None
_cont_lock = asyncio.Lock()


async def get_context_continuity() -> ContextContinuity:
    """Get singleton context continuity tracker."""
    global _continuity
    if _continuity is None:
        async with _cont_lock:
            if _continuity is None:
                history_size = int(os.getenv('JINX_CONTEXT_HISTORY', '20'))
                _continuity = ContextContinuity(history_size=history_size)
    return _continuity


async def analyze_query_continuity(query: str) -> Dict[str, Any]:
    """Analyze query for conversation continuity."""
    cont = await get_context_continuity()
    return await cont.analyze_query(query)


async def get_conversation_context(query: str) -> str:
    """Get conversation context string for query."""
    cont = await get_context_continuity()
    return await cont.get_context_for_query(query)


__all__ = [
    "ContextContinuity",
    "ConversationTurn",
    "get_context_continuity",
    "analyze_query_continuity",
    "get_conversation_context",
]
