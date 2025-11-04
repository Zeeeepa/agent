"""Memory Integration Hub - связывает все системы памяти в единую архитектуру.

Соединяет: Working Memory, Episodic Memory, Knowledge Graph, и Jinx Memory.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class IntegratedMemory:
    """Интегрированное воспоминание из всех источников."""
    content: str
    source: str  # 'working', 'episodic', 'knowledge_graph', 'jinx_memory'
    relevance: float
    importance: float
    timestamp: float
    context: Dict[str, Any]


class MemoryIntegrationHub:
    """Хаб для интеграции всех систем памяти."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._source_weights = {
            'working': 1.0,
            'episodic': 0.9,
            'semantic': 0.8,
            'jinx': 0.7
        }
    
    async def unified_recall(
        self,
        query: str,
        k: int = 5,
        *,
        sources: Optional[List[str]] = None,
        time_decay: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> List[IntegratedMemory]:
        """Unified recall from all memory systems."""
        if sources is None:
            sources = list(self._source_weights.keys())
        
        tasks = []
        if 'working' in sources:
            tasks.append(asyncio.create_task(self._search_working_memory(query)))
        if 'episodic' in sources:
            tasks.append(asyncio.create_task(self._search_episodic_memory(query)))
        if 'semantic' in sources:
            tasks.append(asyncio.create_task(self._search_semantic_memory(query)))
        if 'jinx' in sources:
            tasks.append(asyncio.create_task(self._search_jinx_memory(query)))
        
        memories = []
        
        if not tasks:
            return memories
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            results = []
        except Exception:
            results = []
        
        for result in results:
            if isinstance(result, list):
                memories.extend(result)
        
        await asyncio.sleep(0)
        
        async with self._lock:
            memories.sort(
                key=lambda m: m.relevance * m.importance * self._source_weights.get(m.source, 0.5),
                reverse=True
            )
        
        return memories[:k]
    
    async def _search_working_memory(self, query: str) -> List[IntegratedMemory]:
        """Поиск в рабочей памяти."""
        try:
            from jinx.micro.brain import get_working_memory
            
            wm = await get_working_memory()
            memories = []
            
            query_lower = query.lower()
            
            # Search through working memory items
            for key, item in wm.items.items():
                content_str = str(item.content)
                
                # Simple relevance check
                if query_lower in content_str.lower() or query_lower in key.lower():
                    memories.append(IntegratedMemory(
                        content=f"{key}: {content_str}",
                        source='working',
                        relevance=item.activation,
                        importance=item.priority / 10.0,
                        timestamp=item.last_accessed,
                        context={'key': key, 'access_count': item.access_count}
                    ))
            
            return memories
        except Exception:
            return []
    
    async def _search_episodic_memory(self, query: str) -> List[IntegratedMemory]:
        """Поиск в эпизодической памяти через embeddings."""
        try:
            from jinx.micro.brain import recall_similar_episodes
            
            episodes = await recall_similar_episodes(query, k=5)
            
            memories = []
            for ep in episodes:
                memories.append(IntegratedMemory(
                    content=ep.content,
                    source='episodic',
                    relevance=0.8,  # From semantic search
                    importance=ep.importance,
                    timestamp=ep.timestamp,
                    context={
                        'episode_type': ep.episode_type,
                        'replay_count': ep.replay_count,
                        'emotional_valence': ep.emotional_valence
                    }
                ))
            
            return memories
        except Exception:
            return []
    
    async def _search_knowledge_graph(self, query: str) -> List[IntegratedMemory]:
        """Поиск в графе знаний."""
        try:
            from jinx.micro.brain import query_knowledge_graph
            
            nodes = await query_knowledge_graph(query, query_type='similar')
            
            memories = []
            for node in nodes:
                memories.append(IntegratedMemory(
                    content=str(node.get('data', {})),
                    source='knowledge_graph',
                    relevance=node.get('confidence', 0.5),
                    importance=node.get('confidence', 0.5),
                    timestamp=0.0,
                    context={
                        'node_id': node.get('id'),
                        'node_type': node.get('type')
                    }
                ))
            
            return memories
        except Exception:
            return []
    
    async def _search_jinx_memory(self, query: str) -> List[IntegratedMemory]:
        """Поиск в постоянной памяти Jinx."""
        try:
            from jinx.micro.memory.search import rank_memory
            
            # Safe call with timeout
            try:
                results = await asyncio.wait_for(
                    rank_memory(query, scope='any', k=5, preview_chars=200),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                return []
            except RuntimeError:
                # Event loop closed
                return []
            
            if not results:
                return []
            
            memories = []
            for line in results:
                if line and line.strip():
                    memories.append(IntegratedMemory(
                        content=line.strip(),
                        source='jinx_memory',
                        relevance=0.7,
                        importance=0.6,
                        timestamp=0.0,
                        context={'persistent': True}
                    ))
            
            return memories
        except Exception:
            return []
    
    async def consolidate_memories(self) -> Dict[str, int]:
        """Консолидация: перенос из кратковременной в долговременную память."""
        async with self._lock:
            stats = {'working_to_episodic': 0, 'episodic_to_permanent': 0}
            
            try:
                from jinx.micro.brain import get_working_memory, get_episodic_memory
                
                wm = await get_working_memory()
                em = await get_episodic_memory()
                
                # Transfer important items from working to episodic
                for key, item in list(wm.items.items()):
                    # Criteria: high activation, high priority, multiple accesses
                    if item.activation > 0.7 and item.priority >= 7 and item.access_count >= 3:
                        # Store as episode
                        await em.store_episode(
                            content=f"{key}: {item.content}",
                            episode_type='experience',
                            context={'from_working': True, 'access_count': item.access_count},
                            importance=item.activation * (item.priority / 10.0)
                        )
                        stats['working_to_episodic'] += 1
                
                # Consolidate episodic to permanent
                from jinx.micro.memory.storage import append_evergreen
                
                important_episodes = [
                    ep for ep in em.episodes.values()
                    if ep.importance > 0.8 and not ep.consolidated
                ]
                
                for ep in important_episodes[:10]:  # Limit to prevent flooding
                    # Store in permanent memory
                    await append_evergreen(f"memory: {ep.content}")
                    ep.consolidated = True
                    stats['episodic_to_permanent'] += 1
            
            except Exception:
                pass
            
            return stats
    
    async def link_memories_to_goals(self) -> None:
        """Связать воспоминания с целями."""
        try:
            from jinx.micro.brain import get_goal_system, get_episodic_memory
            
            gs = await get_goal_system()
            em = await get_episodic_memory()
            
            # Get active goals
            active_goals = [
                g for g in gs.goals.values()
                if g.status.value in ('active', 'pending')
            ]
            
            for goal in active_goals:
                # Find relevant episodes
                relevant = await em.recall_similar(goal.description, k=3, min_importance=0.5)
                
                # Store association in goal context
                if relevant and 'related_episodes' not in goal.success_criteria:
                    goal.success_criteria['related_episodes'] = [ep.id for ep in relevant]
        
        except Exception:
            pass
    
    async def enhance_thinking_with_memory(self, query: str) -> Dict[str, Any]:
        """Усилить мышление памятью - интеграция с Meta-Cognitive."""
        try:
            from jinx.micro.brain import think_about, get_meta_cognitive
            
            # Search memories related to query
            memories = await self.unified_recall(query, k=5)
            
            # Build context from memories
            memory_context = {
                'relevant_memories': [
                    {
                        'content': m.content[:200],
                        'source': m.source,
                        'importance': m.importance
                    }
                    for m in memories
                ],
                'memory_count': len(memories)
            }
            
            # Think with memory context
            thought = await think_about(query, memory_context)
            
            # Also reason with evidence from memories
            mc = await get_meta_cognitive()
            evidence = [m.content[:100] for m in memories[:3]]
            
            if evidence:
                conclusion, confidence = await mc.reason_about(query, evidence)
                
                return {
                    'thought': thought.content,
                    'reasoning': conclusion,
                    'confidence': confidence,
                    'memories_used': len(memories)
                }
            
            return {
                'thought': thought.content,
                'memories_used': len(memories)
            }
        
        except Exception:
            return {}
    
    async def create_memory_links(self) -> None:
        """Создать ассоциативные связи между воспоминаниями."""
        try:
            from jinx.micro.brain import get_working_memory, get_episodic_memory, get_knowledge_graph
            
            wm = await get_working_memory()
            em = await get_episodic_memory()
            kg = await get_knowledge_graph()
            
            # Link working memory items
            items = list(wm.items.items())
            for i, (key1, item1) in enumerate(items):
                for key2, item2 in items[i+1:]:
                    # Simple heuristic: link if content overlaps
                    content1 = str(item1.content).lower()
                    content2 = str(item2.content).lower()
                    
                    # Find common words (simple)
                    words1 = set(content1.split())
                    words2 = set(content2.split())
                    overlap = words1 & words2
                    
                    if len(overlap) >= 2:  # At least 2 common words
                        await wm.associate(key1, key2)
            
            # Link episodes to knowledge graph
            for ep in list(em.episodes.values())[:50]:  # Limit processing
                if ep.episode_type in ('learning', 'insight', 'achievement'):
                    # Add to knowledge graph
                    await kg.add_experience(
                        system='episodic_memory',
                        action='learned',
                        outcome=True,
                        context={
                            'content': ep.content[:200],
                            'importance': ep.importance
                        }
                    )
        
        except Exception:
            pass


# Singleton
_memory_hub: Optional[MemoryIntegrationHub] = None
_mh_lock = asyncio.Lock()


async def get_memory_hub() -> MemoryIntegrationHub:
    """Get singleton memory integration hub."""
    global _memory_hub
    if _memory_hub is None:
        async with _mh_lock:
            if _memory_hub is None:
                _memory_hub = MemoryIntegrationHub()
    return _memory_hub


async def search_all_memories(query: str, k: int = 10) -> List[IntegratedMemory]:
    """Search across all memory systems."""
    hub = await get_memory_hub()
    return await hub.unified_recall(query, k)


async def consolidate_all_memories() -> Dict[str, int]:
    """Consolidate memories from short to long term."""
    hub = await get_memory_hub()
    return await hub.consolidate_memories()


async def think_with_memory(query: str) -> Dict[str, Any]:
    """Think about query enhanced with memory search."""
    hub = await get_memory_hub()
    return await hub.enhance_thinking_with_memory(query)


__all__ = [
    "MemoryIntegrationHub",
    "IntegratedMemory",
    "get_memory_hub",
    "search_all_memories",
    "consolidate_all_memories",
    "think_with_memory",
]
