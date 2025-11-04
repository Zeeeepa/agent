"""Concurrent Processor - parallel task processing with intelligence.

Handles multiple user requests simultaneously with context awareness and memory integration.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class TaskPriority(Enum):
    """Task priority levels."""
    URGENT = 0  # User interruption or clarification
    HIGH = 1    # Direct question or code request
    NORMAL = 2  # General query
    LOW = 3     # Background analysis


@dataclass
class ConcurrentTask:
    """Task for concurrent processing."""
    id: str
    query: str
    priority: TaskPriority
    context: Dict[str, Any]
    dependencies: Set[str]  # IDs of tasks this depends on
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    result: Optional[Any]
    error: Optional[str]


@dataclass
class ConversationContext:
    """Tracks conversation flow and continuity."""
    recent_queries: deque[str]
    recent_topics: List[str]
    code_files_mentioned: Set[str]
    active_task_context: Dict[str, Any]
    user_intent_history: List[str]


class ConcurrentProcessor:
    """Process multiple tasks concurrently with intelligence."""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        
        self._tasks: Dict[str, ConcurrentTask] = {}
        self._active: Set[str] = set()
        self._completed: deque[str] = deque(maxlen=100)
        
        self._conversation_context = ConversationContext(
            recent_queries=deque(maxlen=10),
            recent_topics=[],
            code_files_mentioned=set(),
            active_task_context={},
            user_intent_history=[]
        )
        
        self._lock = asyncio.Lock()
        self._task_counter = 0
    
    async def submit_task(
        self,
        query: str,
        *,
        priority: Optional[TaskPriority] = None,
        context: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None
    ) -> str:
        """Submit task for concurrent processing with intelligence."""
        async with self._lock:
            task_id = f"task_{self._task_counter}"
            self._task_counter += 1
            
            # Auto-detect priority if not specified
            if priority is None:
                priority = await self._detect_priority(query)
            
            # Build enriched context
            enriched_context = await self._build_enriched_context(query, context or {})
            
            # Create task
            task = ConcurrentTask(
                id=task_id,
                query=query,
                priority=priority,
                context=enriched_context,
                dependencies=set(depends_on or []),
                created_at=time.time(),
                started_at=None,
                completed_at=None,
                result=None,
                error=None
            )
            
            self._tasks[task_id] = task
            
            # Update conversation context
            self._conversation_context.recent_queries.append(query)
            
            return task_id
    
    async def process_all(self) -> Dict[str, Any]:
        """Process all pending tasks concurrently with intelligence."""
        pending_tasks = []
        
        while True:
            async with self._lock:
                # Get ready tasks (dependencies satisfied, not active)
                ready = await self._get_ready_tasks()
                
                if not ready and not self._active:
                    break  # All done
                
                # Start new tasks up to max_concurrent
                for task_id in ready[:self.max_concurrent - len(self._active)]:
                    self._active.add(task_id)
                    self._tasks[task_id].started_at = time.time()
                    
                    # Create processing task
                    pending_tasks.append(
                        asyncio.create_task(self._process_task(task_id))
                    )
            
            # Wait for at least one to complete
            if pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                pending_tasks = list(pending_tasks)
        
        # Return all results
        return await self._build_unified_result()
    
    async def _detect_priority(self, query: str) -> TaskPriority:
        """Auto-detect task priority using brain intelligence."""
        q_lower = query.lower()
        
        # Urgent: clarification or interruption
        if any(word in q_lower for word in ['wait', 'stop', 'clarify', 'mean', '?']):
            # Check if referencing previous context
            if any(word in q_lower for word in ['previous', 'last', 'above', 'that']):
                return TaskPriority.URGENT
        
        # High: direct code request or specific question
        if any(word in q_lower for word in ['write', 'create', 'fix', 'update', 'change']):
            return TaskPriority.HIGH
        
        if any(word in q_lower for word in ['find', 'show', 'where', 'how']):
            return TaskPriority.HIGH
        
        # Normal: general query
        return TaskPriority.NORMAL
    
    async def _build_enriched_context(
        self,
        query: str,
        base_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build enriched context with memory and intelligence."""
        context = {**base_context}
        
        # Add conversation continuity
        context['conversation_history'] = list(self._conversation_context.recent_queries)
        context['recent_topics'] = self._conversation_context.recent_topics
        
        # Search memories for relevant context
        from jinx.micro.brain import search_all_memories
        memories = await search_all_memories(query, k=5)
        context['relevant_memories'] = [
            {'content': m.content, 'source': m.source, 'importance': m.importance}
            for m in memories
        ]
        
        # Detect if referencing code
        code_files = await self._detect_code_references(query)
        if code_files:
            context['code_files'] = code_files
            self._conversation_context.code_files_mentioned.update(code_files)
        
        # Check for continuation/clarification
        context['is_continuation'] = await self._is_continuation(query)
        context['references_previous'] = await self._references_previous(query)
        
        return context
    
    async def _detect_code_references(self, query: str) -> List[str]:
        """Detect if query references specific code files."""
        # Look for file patterns
        import re
        file_patterns = [
            r'[\w/\\]+\.py',
            r'[\w/\\]+\.js',
            r'[\w/\\]+\.ts',
            r'[\w/\\]+\.go',
        ]
        
        files = []
        for pattern in file_patterns:
            files.extend(re.findall(pattern, query))
        
        return files
    
    async def _is_continuation(self, query: str) -> bool:
        """Check if query is continuation of previous conversation."""
        q_lower = query.lower()
        
        # Check for continuation words
        continuation_words = [
            'also', 'additionally', 'moreover', 'furthermore',
            'and', 'too', 'as well', 'plus'
        ]
        
        return any(word in q_lower for word in continuation_words)
    
    async def _references_previous(self, query: str) -> bool:
        """Check if query references previous context."""
        q_lower = query.lower()
        
        reference_words = [
            'that', 'this', 'it', 'previous', 'last', 'above',
            'earlier', 'before', 'same', 'there'
        ]
        
        return any(word in q_lower for word in reference_words)
    
    async def _get_ready_tasks(self) -> List[str]:
        """Get tasks ready for processing (dependencies satisfied)."""
        ready = []
        
        for task_id, task in self._tasks.items():
            if task.started_at is not None:
                continue  # Already started
            
            if task_id in self._active:
                continue  # Already active
            
            # Check if dependencies are satisfied
            if task.dependencies:
                if not all(dep in self._completed for dep in task.dependencies):
                    continue  # Dependencies not satisfied
            
            ready.append(task_id)
        
        # Sort by priority
        ready.sort(key=lambda tid: self._tasks[tid].priority.value)
        
        return ready
    
    async def _process_task(self, task_id: str) -> None:
        """Process single task with full intelligence."""
        task = self._tasks[task_id]
        
        try:
            # Process with brain orchestrator
            from jinx.micro.brain import process_with_full_intelligence
            
            # Enhance query with context
            enhanced_query = await self._enhance_query_with_context(task)
            
            # Process through full intelligence
            intelligence_result = await process_with_full_intelligence(enhanced_query)
            
            # Execute based on intelligence
            result = await self._execute_with_intelligence(task, intelligence_result)
            
            task.result = result
            task.completed_at = time.time()
            
            # Update conversation context
            await self._update_conversation_context(task, result)
            
            # Store in memory
            await self._store_in_memory(task, result)
            
        except Exception as e:
            task.error = str(e)
        finally:
            async with self._lock:
                self._active.discard(task_id)
                self._completed.append(task_id)
    
    async def _enhance_query_with_context(self, task: ConcurrentTask) -> str:
        """Enhance query with conversation and memory context."""
        enhanced = task.query
        
        # Add conversation context if continuation
        if task.context.get('is_continuation') or task.context.get('references_previous'):
            if self._conversation_context.recent_queries:
                prev = list(self._conversation_context.recent_queries)[-3:]
                context_str = " | Previous context: " + " -> ".join(prev[-2:])
                enhanced = enhanced + context_str
        
        # Add code file context if referenced
        if task.context.get('code_files'):
            files_str = ", ".join(task.context['code_files'])
            enhanced = enhanced + f" | Files: {files_str}"
        
        return enhanced
    
    async def _execute_with_intelligence(
        self,
        task: ConcurrentTask,
        intelligence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task based on intelligence results."""
        route = intelligence.get('route', 'explain')
        intent = intelligence.get('intent', 'general')
        
        result = {
            'route': route,
            'intent': intent,
            'query': task.query,
            'context': task.context
        }
        
        # Execute based on route
        if route == 'code':
            # Code generation or modification
            from jinx.micro.llm import execute_planning_chain_smart
            plan_result = await execute_planning_chain_smart(task.query)
            result['plan'] = plan_result.data
            result['chain_blocks'] = plan_result.data.get('chain_blocks', '')
        
        elif route == 'search':
            # Code search
            from jinx.micro.brain import search_all_memories
            memories = await search_all_memories(task.query, k=10)
            result['search_results'] = [
                {'content': m.content, 'source': m.source}
                for m in memories
            ]
        
        elif route == 'explain':
            # Explanation with context
            result['explanation_context'] = task.context.get('relevant_memories', [])
        
        return result
    
    async def _update_conversation_context(
        self,
        task: ConcurrentTask,
        result: Dict[str, Any]
    ) -> None:
        """Update conversation context after task completion."""
        # Extract topics from result
        if result.get('plan'):
            plan = result['plan']
            if plan.get('goal'):
                self._conversation_context.recent_topics.append(plan['goal'])
        
        # Update active context
        self._conversation_context.active_task_context[task.id] = {
            'query': task.query,
            'result': result,
            'intent': result.get('intent')
        }
    
    async def _store_in_memory(
        self,
        task: ConcurrentTask,
        result: Dict[str, Any]
    ) -> None:
        """Store task and result in long-term memory."""
        from jinx.micro.brain import remember_episode
        
        # Determine importance
        importance = 0.7 if task.priority in (TaskPriority.URGENT, TaskPriority.HIGH) else 0.5
        
        # Store as episode
        await remember_episode(
            content=f"Query: {task.query}. Result: {result.get('intent', 'processed')}",
            episode_type='experience',
            context={
                'priority': task.priority.name,
                'route': result.get('route'),
                'intent': result.get('intent')
            },
            importance=importance
        )
    
    async def _build_unified_result(self) -> Dict[str, Any]:
        """Build unified result from all completed tasks."""
        results = []
        
        for task_id in self._completed:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                results.append({
                    'id': task_id,
                    'query': task.query,
                    'priority': task.priority.name,
                    'result': task.result,
                    'error': task.error,
                    'duration_ms': (
                        (task.completed_at - task.started_at) * 1000
                        if task.started_at and task.completed_at else 0
                    )
                })
        
        return {
            'tasks': results,
            'total': len(results),
            'conversation_context': {
                'recent_queries': list(self._conversation_context.recent_queries),
                'topics': self._conversation_context.recent_topics,
                'code_files': list(self._conversation_context.code_files_mentioned)
            }
        }


# Singleton
_processor: Optional[ConcurrentProcessor] = None
_proc_lock = asyncio.Lock()


async def get_concurrent_processor() -> ConcurrentProcessor:
    """Get singleton concurrent processor."""
    global _processor
    if _processor is None:
        async with _proc_lock:
            if _processor is None:
                max_conc = int(os.getenv('JINX_MAX_CONCURRENT', '3'))
                _processor = ConcurrentProcessor(max_concurrent=max_conc)
    return _processor


async def process_concurrent_queries(queries: List[str]) -> Dict[str, Any]:
    """Process multiple queries concurrently with full intelligence."""
    processor = await get_concurrent_processor()
    
    # Submit all queries
    task_ids = []
    for query in queries:
        task_id = await processor.submit_task(query)
        task_ids.append(task_id)
    
    # Process all
    result = await processor.process_all()
    
    return result


__all__ = [
    "ConcurrentProcessor",
    "TaskPriority",
    "get_concurrent_processor",
    "process_concurrent_queries",
]
