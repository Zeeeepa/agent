"""Episodic Memory - долгосрочная память с embeddings для семантического поиска.

Интегрируется с Working Memory и использует векторные embeddings.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from jinx.micro.embeddings.embed_cache import embed_text_cached as _embed
except Exception:
    _embed = None


@dataclass
class Episode:
    """Эпизод - запись события в долгосрочной памяти."""
    id: str
    content: str
    episode_type: str  # 'experience', 'learning', 'achievement', 'failure', 'insight'
    context: Dict[str, Any]
    embedding: Optional[List[float]]
    importance: float  # 0-1
    emotional_valence: float  # -1 (negative) to 1 (positive)
    timestamp: float
    consolidated: bool  # Moved to long-term
    replay_count: int  # How many times recalled


class EpisodicMemory:
    """Долгосрочная эпизодическая память с семантическим поиском."""
    
    def __init__(self, state_path: str = "log/episodic_memory.json"):
        self.state_path = state_path
        
        # Episode storage
        self.episodes: Dict[str, Episode] = {}
        
        # Recent episodes buffer (for consolidation)
        self.recent_buffer: deque[str] = deque(maxlen=50)
        
        # Episode counter
        self._episode_counter = 0
        
        # Consolidation threshold
        self.consolidation_threshold = 0.7  # importance threshold
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load memory state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore episodes
                for ep_data in data.get('episodes', []):
                    episode = Episode(
                        id=ep_data['id'],
                        content=ep_data['content'],
                        episode_type=ep_data['episode_type'],
                        context=ep_data['context'],
                        embedding=ep_data.get('embedding'),
                        importance=ep_data['importance'],
                        emotional_valence=ep_data['emotional_valence'],
                        timestamp=ep_data['timestamp'],
                        consolidated=ep_data['consolidated'],
                        replay_count=ep_data['replay_count']
                    )
                    self.episodes[episode.id] = episode
                
                self._episode_counter = data.get('counter', 0)
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist memory state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize episodes (limit to most important)
                important_episodes = sorted(
                    self.episodes.values(),
                    key=lambda e: e.importance * (1 + e.replay_count),
                    reverse=True
                )[:1000]  # Keep top 1000
                
                episodes_data = [
                    {
                        'id': ep.id,
                        'content': ep.content,
                        'episode_type': ep.episode_type,
                        'context': ep.context,
                        'embedding': ep.embedding,
                        'importance': ep.importance,
                        'emotional_valence': ep.emotional_valence,
                        'timestamp': ep.timestamp,
                        'consolidated': ep.consolidated,
                        'replay_count': ep.replay_count
                    }
                    for ep in important_episodes
                ]
                
                data = {
                    'episodes': episodes_data,
                    'counter': self._episode_counter,
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def store_episode(
        self,
        content: str,
        episode_type: str,
        context: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        emotional_valence: float = 0.0
    ) -> str:
        """Сохранить эпизод в память."""
        async with self._lock:
            episode_id = f"ep_{self._episode_counter}"
            self._episode_counter += 1
            
            # Compute embedding if available
            embedding = None
            if _embed:
                try:
                    embedding = await _embed(content[:512])
                except Exception:
                    pass
            
            # Auto-compute importance if not provided
            if importance is None:
                importance = await self._compute_importance(content, episode_type, context or {})
            
            episode = Episode(
                id=episode_id,
                content=content,
                episode_type=episode_type,
                context=context or {},
                embedding=embedding,
                importance=importance,
                emotional_valence=emotional_valence,
                timestamp=time.time(),
                consolidated=False,
                replay_count=0
            )
            
            self.episodes[episode_id] = episode
            self.recent_buffer.append(episode_id)
            
            # Trigger consolidation if needed
            if len(self.recent_buffer) >= 20:
                await self._consolidate()
            
            return episode_id
    
    async def _compute_importance(
        self,
        content: str,
        episode_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Вычислить важность эпизода."""
        importance = 0.5  # Base
        
        # Type-based importance
        type_weights = {
            'achievement': 0.9,
            'insight': 0.85,
            'learning': 0.8,
            'failure': 0.75,
            'experience': 0.6
        }
        importance = type_weights.get(episode_type, 0.5)
        
        # Context-based adjustments
        if context.get('success'):
            importance += 0.1
        if context.get('novel'):  # Novel experience
            importance += 0.15
        if context.get('critical'):
            importance += 0.2
        
        # Content-based (keywords)
        content_lower = content.lower()
        if any(word in content_lower for word in ['discovered', 'learned', 'realized']):
            importance += 0.1
        
        return min(1.0, importance)
    
    async def recall_similar(
        self,
        query: str,
        k: int = 5,
        min_importance: float = 0.3
    ) -> List[Episode]:
        """Вспомнить похожие эпизоды через semantic search."""
        async with self._lock:
            if not _embed or not self.episodes:
                return []
            
            # Get query embedding
            try:
                query_emb = await _embed(query[:512])
            except Exception:
                return []
            
            if not query_emb:
                return []
            
            # Compute similarities
            scored_episodes: List[Tuple[Episode, float]] = []
            
            for episode in self.episodes.values():
                if not episode.embedding:
                    continue
                
                if episode.importance < min_importance:
                    continue
                
                # Cosine similarity
                similarity = self._cosine_similarity(query_emb, episode.embedding)
                
                # Boost by importance and replay count
                score = similarity * (1 + episode.importance * 0.5) * (1 + episode.replay_count * 0.1)
                
                scored_episodes.append((episode, score))
            
            # Sort and return top k
            scored_episodes.sort(key=lambda x: x[1], reverse=True)
            
            # Increment replay count
            results = []
            for episode, score in scored_episodes[:k]:
                episode.replay_count += 1
                results.append(episode)
            
            return results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity."""
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
    
    async def _consolidate(self) -> None:
        """Консолидация: переместить важные эпизоды в долгосрочную память."""
        # Get recent episodes
        recent_ids = list(self.recent_buffer)
        
        for ep_id in recent_ids:
            if ep_id not in self.episodes:
                continue
            
            episode = self.episodes[ep_id]
            
            # Consolidate if important enough
            if episode.importance >= self.consolidation_threshold:
                episode.consolidated = True
                
                # Also store in Jinx's permanent memory
                try:
                    from jinx.micro.memory.storage import append_compact
                    await append_compact(f"MEMORY: {episode.content}")
                except Exception:
                    pass
        
        # Save state
        await self._save_state()
    
    async def replay_episode(self, episode_id: str) -> Optional[Episode]:
        """Replay (recall) specific episode - strengthens memory."""
        async with self._lock:
            if episode_id not in self.episodes:
                return None
            
            episode = self.episodes[episode_id]
            episode.replay_count += 1
            episode.importance = min(1.0, episode.importance * 1.05)
            
            return episode
    
    async def get_by_type(self, episode_type: str, limit: int = 10) -> List[Episode]:
        """Получить эпизоды по типу."""
        async with self._lock:
            episodes = [
                ep for ep in self.episodes.values()
                if ep.episode_type == episode_type
            ]
            
            # Sort by importance and recency
            episodes.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
            
            return episodes[:limit]
    
    async def get_emotional_memories(
        self,
        valence: str = 'positive',
        limit: int = 10
    ) -> List[Episode]:
        """Получить эмоционально окрашенные воспоминания."""
        async with self._lock:
            if valence == 'positive':
                episodes = [ep for ep in self.episodes.values() if ep.emotional_valence > 0.3]
            elif valence == 'negative':
                episodes = [ep for ep in self.episodes.values() if ep.emotional_valence < -0.3]
            else:
                episodes = list(self.episodes.values())
            
            episodes.sort(key=lambda e: abs(e.emotional_valence), reverse=True)
            
            return episodes[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику памяти."""
        if not self.episodes:
            return {'total': 0}
        
        type_counts = {}
        for ep in self.episodes.values():
            type_counts[ep.episode_type] = type_counts.get(ep.episode_type, 0) + 1
        
        consolidated_count = sum(1 for ep in self.episodes.values() if ep.consolidated)
        
        return {
            'total_episodes': len(self.episodes),
            'consolidated': consolidated_count,
            'types': type_counts,
            'avg_importance': sum(ep.importance for ep in self.episodes.values()) / len(self.episodes),
            'most_replayed': max((ep.replay_count for ep in self.episodes.values()), default=0)
        }


# Singleton
_episodic_memory: Optional[EpisodicMemory] = None
_em_lock = asyncio.Lock()


async def get_episodic_memory() -> EpisodicMemory:
    """Get singleton episodic memory."""
    global _episodic_memory
    if _episodic_memory is None:
        async with _em_lock:
            if _episodic_memory is None:
                _episodic_memory = EpisodicMemory()
    return _episodic_memory


async def remember_episode(
    content: str,
    episode_type: str,
    context: Optional[Dict[str, Any]] = None,
    importance: Optional[float] = None
) -> str:
    """Store episode in long-term memory."""
    em = await get_episodic_memory()
    return await em.store_episode(content, episode_type, context, importance)


async def recall_similar_episodes(query: str, k: int = 5) -> List[Episode]:
    """Recall similar episodes via semantic search."""
    em = await get_episodic_memory()
    return await em.recall_similar(query, k)


async def integrate_with_working_memory() -> None:
    """Интегрировать рабочую память с эпизодической."""
    from jinx.micro.brain import get_working_memory
    
    wm = await get_working_memory()
    em = await get_episodic_memory()
    
    # Transfer important items from working to episodic
    for key, item in wm.items.items():
        if item.importance > 0.7 and item.access_count > 3:
            # This is important - store as episode
            await em.store_episode(
                content=f"{key}: {item.content}",
                episode_type='experience',
                context={'from_working_memory': True},
                importance=item.importance
            )


__all__ = [
    "EpisodicMemory",
    "Episode",
    "get_episodic_memory",
    "remember_episode",
    "recall_similar_episodes",
    "integrate_with_working_memory",
]
