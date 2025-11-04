"""Working Memory - активная рабочая память AI для текущих задач.

Краткосрочная память для активной обработки информации.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class MemoryItem:
    """Элемент рабочей памяти."""
    content: Any
    item_type: str
    priority: int
    activation: float  # 0-1, decreases over time
    created_at: float
    last_accessed: float
    access_count: int
    associations: Set[str]  # Links to other items


class WorkingMemory:
    """Рабочая память для текущей обработки - как оперативная память мозга."""
    
    def __init__(self, capacity: int = 7):  # Miller's Law: 7±2 items
        self.capacity = capacity
        self.items: Dict[str, MemoryItem] = {}
        self.activation_decay = 0.95  # Per second
        
        # Attention spotlight
        self.current_focus: Optional[str] = None
        
        # Access patterns
        self.access_history: deque[str] = deque(maxlen=100)
        
        self._lock = asyncio.Lock()
        self._last_decay = time.time()
    
    async def store(
        self,
        key: str,
        content: Any,
        item_type: str = 'fact',
        priority: int = 1
    ) -> bool:
        """Сохранить в рабочую память."""
        async with self._lock:
            # Decay existing items
            await self._decay_activation()
            
            # Check capacity
            if len(self.items) >= self.capacity and key not in self.items:
                # Remove least activated item
                await self._evict_least_activated()
            
            now = time.time()
            
            if key in self.items:
                # Update existing
                item = self.items[key]
                item.content = content
                item.activation = min(1.0, item.activation + 0.3)
                item.last_accessed = now
                item.access_count += 1
            else:
                # Create new
                self.items[key] = MemoryItem(
                    content=content,
                    item_type=item_type,
                    priority=priority,
                    activation=1.0,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    associations=set()
                )
            
            return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Получить из рабочей памяти."""
        async with self._lock:
            if key not in self.items:
                return None
            
            item = self.items[key]
            
            # Boost activation on access
            item.activation = min(1.0, item.activation + 0.2)
            item.last_accessed = time.time()
            item.access_count += 1
            
            # Update focus
            self.current_focus = key
            
            # Record access
            self.access_history.append(key)
            
            return item.content
    
    async def associate(self, key1: str, key2: str) -> None:
        """Создать ассоциацию между элементами."""
        async with self._lock:
            if key1 in self.items:
                self.items[key1].associations.add(key2)
            if key2 in self.items:
                self.items[key2].associations.add(key1)
    
    async def recall_associated(self, key: str) -> List[Any]:
        """Вспомнить ассоциированные элементы."""
        async with self._lock:
            if key not in self.items:
                return []
            
            item = self.items[key]
            associated = []
            
            for assoc_key in item.associations:
                if assoc_key in self.items:
                    associated.append(self.items[assoc_key].content)
            
            return associated
    
    async def get_focused(self) -> Optional[Any]:
        """Получить текущий focus of attention."""
        if self.current_focus and self.current_focus in self.items:
            return await self.retrieve(self.current_focus)
        return None
    
    async def _decay_activation(self) -> None:
        """Уменьшить activation всех элементов (забывание)."""
        now = time.time()
        elapsed = now - self._last_decay
        
        if elapsed < 1.0:
            return
        
        decay_factor = self.activation_decay ** elapsed
        
        to_remove = []
        for key, item in self.items.items():
            item.activation *= decay_factor
            
            # Remove if activation too low
            if item.activation < 0.1:
                to_remove.append(key)
        
        for key in to_remove:
            del self.items[key]
        
        self._last_decay = now
    
    async def _evict_least_activated(self) -> None:
        """Удалить элемент с наименьшей активацией."""
        if not self.items:
            return
        
        # Find item with lowest activation and priority
        min_key = min(
            self.items.keys(),
            key=lambda k: (self.items[k].activation, self.items[k].priority)
        )
        
        del self.items[min_key]
    
    async def clear(self) -> None:
        """Очистить рабочую память."""
        async with self._lock:
            self.items.clear()
            self.current_focus = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        if not self.items:
            return {'capacity': self.capacity, 'used': 0}
        
        return {
            'capacity': self.capacity,
            'used': len(self.items),
            'avg_activation': sum(i.activation for i in self.items.values()) / len(self.items),
            'current_focus': self.current_focus,
            'total_accesses': sum(i.access_count for i in self.items.values())
        }


# Singleton
_working_memory: Optional[WorkingMemory] = None
_wm_lock = asyncio.Lock()


async def get_working_memory() -> WorkingMemory:
    """Get singleton working memory."""
    global _working_memory
    if _working_memory is None:
        async with _wm_lock:
            if _working_memory is None:
                _working_memory = WorkingMemory()
    return _working_memory


async def remember(key: str, content: Any, priority: int = 1) -> bool:
    """Store in working memory."""
    wm = await get_working_memory()
    return await wm.store(key, content, priority=priority)


async def recall(key: str) -> Optional[Any]:
    """Retrieve from working memory."""
    wm = await get_working_memory()
    return await wm.retrieve(key)


async def get_focus() -> Optional[Any]:
    """Get current focus of attention."""
    wm = await get_working_memory()
    return await wm.get_focused()


__all__ = [
    "WorkingMemory",
    "MemoryItem",
    "get_working_memory",
    "remember",
    "recall",
    "get_focus",
]
