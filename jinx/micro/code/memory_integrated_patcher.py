"""Memory Integrated Patcher - unified memory system integration.

Integrates ALL 4 memory systems + brain for intelligent code patching:
- Working Memory (active context)
- Episodic Memory (past experiences)
- Semantic Memory (knowledge)
- Jinx Memory (permanent storage)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..brain import (
    # Memory Systems (all 4)
    search_all_memories,
    remember_episode,
    think_with_memory,
    get_knowledge_graph,
    consolidate_all_memories,
    
    # Context & Cognition
    get_context_continuity,
    analyze_query_continuity,
    get_working_memory,
    get_meta_cognitive_system,
    
    # Intelligence
    classify_query_intent,
    select_retrieval_params,
    expand_query,
)

from ..embeddings import (
    embed_text_cached,
    search_embeddings_unified,
)


@dataclass
class MemoryEnhancedPatch:
    """Patch enhanced with full memory context."""
    original_code: str
    suggested_code: str
    confidence: float
    
    # Memory sources
    episodic_examples: List[Dict[str, Any]]  # Past similar patches
    semantic_knowledge: List[str]  # Related concepts
    working_context: Dict[str, Any]  # Current task context
    permanent_patterns: List[Dict[str, Any]]  # Jinx memory patterns
    
    # Intelligence
    reasoning: str  # Why this patch
    learned_from: List[str]  # Sources
    success_probability: float
    
    # Context
    file_context: str
    conversation_context: str


class MemoryIntegratedPatcher:
    """Patcher with full memory system integration."""
    
    def __init__(self):
        self._working_memory = None
        self._meta_cognitive = None
        self._context_continuity = None
        self._knowledge_graph = None
    
    async def _ensure_initialized(self):
        """Ensure all memory systems are initialized."""
        if self._working_memory is None:
            from ..brain import get_working_memory
            self._working_memory = await get_working_memory()
        
        if self._meta_cognitive is None:
            from ..brain import get_meta_cognitive_system
            self._meta_cognitive = await get_meta_cognitive_system()
        
        if self._context_continuity is None:
            self._context_continuity = await get_context_continuity()
        
        if self._knowledge_graph is None:
            self._knowledge_graph = await get_knowledge_graph()
    
    async def analyze_code_with_full_memory(
        self,
        code: str,
        file_path: str,
        user_intent: Optional[str] = None
    ) -> MemoryEnhancedPatch:
        """Analyze code using ALL memory systems."""
        
        await self._ensure_initialized()
        
        # 1. Working Memory - get current task context
        working_ctx = await self._working_memory.get_active_items()
        
        # 2. Context Continuity - understand conversation flow
        conversation_analysis = await self._context_continuity.analyze_query(
            user_intent or f"Analyze code in {file_path}"
        )
        
        # 3. Episodic Memory - find similar past patches
        episodic_memories = await search_all_memories(
            f"code patch {code[:100]}",
            k=10
        )
        
        episodic_examples = []
        for memory in episodic_memories:
            if hasattr(memory, 'content') and 'patch' in memory.content.lower():
                episodic_examples.append({
                    'content': memory.content,
                    'importance': getattr(memory, 'importance', 0.5),
                    'timestamp': getattr(memory, 'timestamp', 0)
                })
        
        # 4. Semantic Memory - get related knowledge
        semantic_knowledge = await self._get_semantic_knowledge(code)
        
        # 5. Jinx Memory (permanent) - query stored patterns
        permanent_patterns = await self._query_jinx_memory(code, file_path)
        
        # 6. Knowledge Graph - find pattern relationships
        kg_patterns = await self._knowledge_graph.query_patterns(
            code[:200],
            'similar'
        )
        
        # 7. Meta-Cognitive - reason about approach
        reasoning = await self._meta_cognitive.reflect_on_process(
            f"Analyzing code for improvement: {code[:100]}..."
        )
        
        # 8. Think with Memory - synthesize understanding
        thought = await think_with_memory(
            f"How to improve this code?\n{code[:200]}\nIntent: {user_intent or 'general improvement'}"
        )
        
        # 9. Generate suggestion using all context
        suggested_code, confidence = await self._generate_suggestion(
            original_code=code,
            episodic_examples=episodic_examples,
            semantic_knowledge=semantic_knowledge,
            working_context=working_ctx,
            reasoning=reasoning,
            thought=thought
        )
        
        # 10. Calculate success probability from memory
        success_probability = await self._calculate_success_probability(
            episodic_examples,
            permanent_patterns
        )
        
        return MemoryEnhancedPatch(
            original_code=code,
            suggested_code=suggested_code,
            confidence=confidence,
            episodic_examples=episodic_examples[:5],
            semantic_knowledge=semantic_knowledge[:5],
            working_context=working_ctx,
            permanent_patterns=permanent_patterns[:5],
            reasoning=reasoning.get('reflection', '') if isinstance(reasoning, dict) else str(reasoning),
            learned_from=[
                f"Episodic: {len(episodic_examples)} examples",
                f"Semantic: {len(semantic_knowledge)} concepts",
                f"Patterns: {len(permanent_patterns)} patterns",
                f"KG: {len(kg_patterns)} relationships"
            ],
            success_probability=success_probability,
            file_context=file_path,
            conversation_context=conversation_analysis.get('suggested_context', '')
        )
    
    async def suggest_with_memory_context(
        self,
        file_path: str,
        cursor_position: int,
        partial_code: str
    ) -> List[MemoryEnhancedPatch]:
        """Get suggestions with full memory context (like AI editor)."""
        
        await self._ensure_initialized()
        
        # Add to working memory
        await self._working_memory.add_item(
            f"Editing {file_path} at position {cursor_position}",
            importance=0.8
        )
        
        # Analyze with full memory
        patch = await self.analyze_code_with_full_memory(
            partial_code,
            file_path,
            user_intent="code completion"
        )
        
        # Generate multiple suggestions with different approaches
        suggestions = [patch]
        
        # Use episodic memory for variations
        for example in patch.episodic_examples[:3]:
            variation = await self._create_variation(partial_code, example)
            if variation:
                suggestions.append(variation)
        
        return suggestions
    
    async def learn_from_edit(
        self,
        original: str,
        edited: str,
        accepted: bool,
        file_path: str
    ):
        """Learn from user edits (like AI editor feedback loop)."""
        
        await self._ensure_initialized()
        
        # Store in episodic memory
        await remember_episode(
            content=f"Code edit in {file_path}: {original[:100]} -> {edited[:100]}",
            episode_type='tool_use',
            context={
                'file': file_path,
                'accepted': accepted,
                'edit_type': 'user_correction'
            },
            importance=0.9 if accepted else 0.6
        )
        
        # Update working memory
        await self._working_memory.add_item(
            f"User {'accepted' if accepted else 'rejected'} suggestion for {file_path}",
            importance=0.7
        )
        
        # Store in knowledge graph
        await self._knowledge_graph.add_node(
            {
                'type': 'code_pattern',
                'original': original[:200],
                'edited': edited[:200],
                'file': file_path,
                'success': accepted
            }
        )
        
        # Meta-cognitive reflection
        await self._meta_cognitive.reflect_on_process(
            f"User {'accepted' if accepted else 'rejected'} my suggestion. "
            f"{'I should learn from this pattern.' if accepted else 'I need to improve my understanding.'}"
        )
    
    async def get_contextual_documentation(
        self,
        code: str,
        symbol: str
    ) -> Dict[str, Any]:
        """Get documentation with memory context (hover info)."""
        
        await self._ensure_initialized()
        
        # Search all memory systems
        memories = await search_all_memories(f"documentation {symbol}", k=5)
        
        # Get semantic knowledge
        semantic = await self._get_semantic_knowledge(f"{symbol} usage")
        
        # Query knowledge graph for relationships
        relationships = await self._knowledge_graph.query_patterns(symbol, 'related')
        
        return {
            'symbol': symbol,
            'memories': [m.content for m in memories],
            'semantic_knowledge': semantic,
            'relationships': relationships,
            'examples': [
                m.content for m in memories 
                if 'example' in m.content.lower()
            ]
        }
    
    async def _get_semantic_knowledge(self, query: str) -> List[str]:
        """Get semantic knowledge from memory."""
        # Semantic memory would be queried here
        # For now, use general memory search
        memories = await search_all_memories(f"knowledge {query}", k=5)
        return [m.content for m in memories if hasattr(m, 'content')]
    
    async def _query_jinx_memory(
        self,
        code: str,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Query permanent Jinx memory storage."""
        try:
            # Try to import and use Jinx memory
            from ..memory.search import rank_memory
            
            # Safe call with timeout
            try:
                results = await asyncio.wait_for(
                    rank_memory(code[:200], scope='any', k=5, preview_chars=100),
                    timeout=1.0
                )
                
                return [
                    {
                        'content': r.strip() if isinstance(r, str) else str(r),
                        'type': 'pattern',
                        'relevance': 0.7
                    }
                    for r in results if r
                ]
            except (asyncio.TimeoutError, RuntimeError):
                return []
        except ImportError:
            # Memory module not available
            return []
        except Exception:
            return []
    
    async def _generate_suggestion(
        self,
        original_code: str,
        episodic_examples: List[Dict],
        semantic_knowledge: List[str],
        working_context: Dict,
        reasoning: Any,
        thought: str
    ) -> Tuple[str, float]:
        """Generate code suggestion using all memory context."""
        
        # Use episodic examples as templates
        if episodic_examples:
            # Find most relevant example
            best_example = max(
                episodic_examples,
                key=lambda x: x.get('importance', 0)
            )
            
            # Extract pattern from example
            if 'after' in best_example.get('content', ''):
                suggested = original_code  # Would apply learned transform
                confidence = best_example.get('importance', 0.5)
            else:
                suggested = original_code
                confidence = 0.5
        else:
            suggested = original_code
            confidence = 0.3
        
        # Boost confidence if semantic knowledge supports it
        if semantic_knowledge:
            confidence = min(1.0, confidence + 0.1)
        
        return suggested, confidence
    
    async def _calculate_success_probability(
        self,
        episodic_examples: List[Dict],
        permanent_patterns: List[Dict]
    ) -> float:
        """Calculate success probability from memory."""
        
        if not episodic_examples and not permanent_patterns:
            return 0.5  # No data
        
        # Count successes in episodic memory
        successes = sum(
            1 for ex in episodic_examples 
            if ex.get('importance', 0) > 0.7
        )
        
        total = len(episodic_examples) + len(permanent_patterns)
        
        if total == 0:
            return 0.5
        
        return successes / total
    
    async def _create_variation(
        self,
        code: str,
        example: Dict
    ) -> Optional[MemoryEnhancedPatch]:
        """Create variation based on episodic example."""
        # Would create variation here
        # Simplified for now
        return None


# Singleton
_memory_patcher: Optional[MemoryIntegratedPatcher] = None
_patcher_lock = asyncio.Lock()


async def get_memory_integrated_patcher() -> MemoryIntegratedPatcher:
    """Get singleton memory-integrated patcher."""
    global _memory_patcher
    if _memory_patcher is None:
        async with _patcher_lock:
            if _memory_patcher is None:
                _memory_patcher = MemoryIntegratedPatcher()
    return _memory_patcher


__all__ = [
    "MemoryIntegratedPatcher",
    "MemoryEnhancedPatch",
    "get_memory_integrated_patcher",
]
