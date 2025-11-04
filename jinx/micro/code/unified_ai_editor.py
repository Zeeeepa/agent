"""Unified AI Editor - complete modern AI editor capabilities.

Combines ALL systems:
- Memory integration (4 systems)
- AI code analysis
- Semantic patching
- AST transforms
- Real-time intelligence
- Brain-powered suggestions
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .memory_integrated_patcher import (
    get_memory_integrated_patcher,
    MemoryEnhancedPatch
)
from .ai_code_analyzer import (
    get_ai_analyzer,
    CodeDiagnostic,
    CodeSuggestion,
    CodeAction
)
from .semantic_patcher import get_semantic_patcher
from .ast_transformer import get_ast_engine
from .patch_orchestrator import get_patch_orchestrator

from ..brain import (
    get_concurrent_processor,
    get_context_continuity,
    analyze_query_continuity,
    think_with_memory,
)


@dataclass
class EditorContext:
    """Complete editor context."""
    file_path: str
    content: str
    cursor_line: int
    cursor_column: int
    selection_start: Optional[tuple[int, int]]
    selection_end: Optional[tuple[int, int]]
    
    # Memory context
    conversation_context: str
    working_memory: List[str]
    recent_edits: List[Dict[str, Any]]
    
    # Intelligence
    diagnostics: List[CodeDiagnostic]
    suggestions: List[CodeSuggestion]
    actions: List[CodeAction]


@dataclass
class IntelligentEdit:
    """AI-suggested edit with full context."""
    title: str
    description: str
    confidence: float
    
    # Edit details
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    new_text: str
    
    # Intelligence
    reasoning: str
    learned_from: List[str]
    memory_sources: List[str]
    
    # Preview
    preview_before: str
    preview_after: str


class UnifiedAIEditor:
    """Complete AI editor with full brain integration."""
    
    def __init__(self):
        self._memory_patcher = None
        self._ai_analyzer = None
        self._semantic_patcher = None
        self._ast_engine = None
        self._orchestrator = None
        self._concurrent_processor = None
        self._context_continuity = None
        
        self._active_contexts: Dict[str, EditorContext] = {}
        self._lock = asyncio.Lock()
    
    async def _ensure_initialized(self):
        """Initialize all systems."""
        if self._memory_patcher is None:
            self._memory_patcher = await get_memory_integrated_patcher()
        if self._ai_analyzer is None:
            self._ai_analyzer = await get_ai_analyzer()
        if self._semantic_patcher is None:
            self._semantic_patcher = await get_semantic_patcher()
        if self._ast_engine is None:
            self._ast_engine = await get_ast_engine()
        if self._orchestrator is None:
            self._orchestrator = await get_patch_orchestrator()
        if self._concurrent_processor is None:
            self._concurrent_processor = await get_concurrent_processor()
        if self._context_continuity is None:
            self._context_continuity = await get_context_continuity()
    
    async def open_file(
        self,
        file_path: str,
        content: str
    ) -> EditorContext:
        """Open file in AI editor."""
        
        await self._ensure_initialized()
        
        # Analyze file comprehensively
        analysis = await self._ai_analyzer.analyze_file(file_path, content)
        
        # Create context
        context = EditorContext(
            file_path=file_path,
            content=content,
            cursor_line=0,
            cursor_column=0,
            selection_start=None,
            selection_end=None,
            conversation_context='',
            working_memory=[],
            recent_edits=[],
            diagnostics=analysis['diagnostics'],
            suggestions=[],
            actions=[]
        )
        
        # Store context
        async with self._lock:
            self._active_contexts[file_path] = context
        
        return context
    
    async def get_intelligent_suggestions(
        self,
        file_path: str,
        line: int,
        column: int
    ) -> List[IntelligentEdit]:
        """Get AI suggestions at cursor (like Copilot/Cursor)."""
        
        await self._ensure_initialized()
        
        context = await self._get_context(file_path)
        if not context:
            return []
        
        # Update cursor position
        context.cursor_line = line
        context.cursor_column = column
        
        # Get completions from AI analyzer
        completions = await self._ai_analyzer.get_completions(
            file_path, context.content, line, column
        )
        
        # Get memory-enhanced suggestions
        memory_patches = await self._memory_patcher.suggest_with_memory_context(
            file_path, line * 100 + column, context.content
        )
        
        # Get semantic suggestions
        semantic_suggestions = await self._semantic_patcher.suggest_patches_for_file(
            file_path, context.content
        )
        
        # Combine all suggestions
        intelligent_edits = []
        
        # From completions
        for comp in completions[:5]:
            intelligent_edits.append(IntelligentEdit(
                title=comp.label,
                description=comp.detail,
                confidence=comp.confidence,
                start_line=line,
                start_column=column,
                end_line=line,
                end_column=column,
                new_text=comp.insert_text,
                reasoning=comp.documentation,
                learned_from=['AI Analyzer', 'Memory'] if comp.learned else ['AI Analyzer'],
                memory_sources=[],
                preview_before=context.content.split('\n')[line] if line < len(context.content.split('\n')) else '',
                preview_after=comp.insert_text
            ))
        
        # From memory patches
        for patch in memory_patches[:3]:
            intelligent_edits.append(IntelligentEdit(
                title="Memory-based improvement",
                description=patch.reasoning,
                confidence=patch.confidence,
                start_line=line,
                start_column=0,
                end_line=line + len(patch.original_code.split('\n')),
                end_column=0,
                new_text=patch.suggested_code,
                reasoning=patch.reasoning,
                learned_from=patch.learned_from,
                memory_sources=[
                    f"Episodic: {len(patch.episodic_examples)}",
                    f"Semantic: {len(patch.semantic_knowledge)}",
                    f"Patterns: {len(patch.permanent_patterns)}"
                ],
                preview_before=patch.original_code,
                preview_after=patch.suggested_code
            ))
        
        # Sort by confidence
        intelligent_edits.sort(key=lambda e: e.confidence, reverse=True)
        
        return intelligent_edits[:10]
    
    async def apply_edit_with_learning(
        self,
        file_path: str,
        edit: IntelligentEdit,
        accepted: bool
    ):
        """Apply edit and learn from feedback."""
        
        await self._ensure_initialized()
        
        context = await self._get_context(file_path)
        if not context:
            return
        
        # Apply edit if accepted
        if accepted:
            lines = context.content.split('\n')
            
            # Replace lines
            new_lines = (
                lines[:edit.start_line] +
                [edit.new_text] +
                lines[edit.end_line + 1:]
            )
            
            context.content = '\n'.join(new_lines)
            
            # Record edit
            context.recent_edits.append({
                'edit': edit,
                'accepted': True,
                'timestamp': asyncio.get_event_loop().time()
            })
        
        # Learn from feedback
        await self._memory_patcher.learn_from_edit(
            original=edit.preview_before,
            edited=edit.preview_after if accepted else edit.preview_before,
            accepted=accepted,
            file_path=file_path
        )
        
        # Record in context continuity
        await self._context_continuity.record_turn(
            user_query=f"Edit at {file_path}:{edit.start_line}",
            system_response=edit.description,
            intent='code_edit',
            code_files={file_path},
            topics=[edit.title]
        )
    
    async def get_refactoring_suggestions(
        self,
        file_path: str,
        selection_start: tuple[int, int],
        selection_end: tuple[int, int]
    ) -> List[IntelligentEdit]:
        """Get refactoring suggestions for selection."""
        
        await self._ensure_initialized()
        
        context = await self._get_context(file_path)
        if not context:
            return []
        
        # Extract selected text
        lines = context.content.split('\n')
        start_line, start_col = selection_start
        end_line, end_col = selection_end
        
        if start_line == end_line:
            selected = lines[start_line][start_col:end_col]
        else:
            selected_lines = [lines[start_line][start_col:]]
            selected_lines.extend(lines[start_line + 1:end_line])
            selected_lines.append(lines[end_line][:end_col])
            selected = '\n'.join(selected_lines)
        
        # Get refactoring suggestions
        refactorings = []
        
        # AST-based refactoring
        try:
            # Add type hints
            with_hints = await self._ast_engine.add_type_hints(selected)
            if with_hints != selected:
                refactorings.append(IntelligentEdit(
                    title="Add type hints",
                    description="Add type annotations to improve code clarity",
                    confidence=0.9,
                    start_line=start_line,
                    start_column=start_col,
                    end_line=end_line,
                    end_column=end_col,
                    new_text=with_hints,
                    reasoning="Type hints improve code documentation and enable better IDE support",
                    learned_from=['AST Engine'],
                    memory_sources=[],
                    preview_before=selected,
                    preview_after=with_hints
                ))
            
            # Convert to async
            as_async = await self._ast_engine.convert_to_async(selected)
            if as_async != selected:
                refactorings.append(IntelligentEdit(
                    title="Convert to async",
                    description="Convert synchronous code to asynchronous",
                    confidence=0.85,
                    start_line=start_line,
                    start_column=start_col,
                    end_line=end_line,
                    end_column=end_col,
                    new_text=as_async,
                    reasoning="Async code enables better concurrency and performance",
                    learned_from=['AST Engine'],
                    memory_sources=[],
                    preview_before=selected,
                    preview_after=as_async
                ))
        except Exception:
            pass
        
        # Memory-based refactoring
        memory_patch = await self._memory_patcher.analyze_code_with_full_memory(
            selected, file_path, user_intent="refactor"
        )
        
        if memory_patch.confidence > 0.7:
            refactorings.append(IntelligentEdit(
                title="Memory-based refactoring",
                description=memory_patch.reasoning,
                confidence=memory_patch.confidence,
                start_line=start_line,
                start_column=start_col,
                end_line=end_line,
                end_column=end_col,
                new_text=memory_patch.suggested_code,
                reasoning=memory_patch.reasoning,
                learned_from=memory_patch.learned_from,
                memory_sources=[
                    f"Episodic: {len(memory_patch.episodic_examples)}",
                    f"Semantic: {len(memory_patch.semantic_knowledge)}"
                ],
                preview_before=memory_patch.original_code,
                preview_after=memory_patch.suggested_code
            ))
        
        return refactorings
    
    async def chat_with_codebase(
        self,
        query: str,
        context_files: List[str]
    ) -> str:
        """Chat about codebase (like Cursor chat)."""
        
        await self._ensure_initialized()
        
        # Analyze query continuity
        analysis = await analyze_query_continuity(query)
        
        # Think with memory
        response = await think_with_memory(
            f"User asks: {query}\nContext files: {', '.join(context_files)}\n"
            f"Conversation context: {analysis.get('suggested_context', '')}"
        )
        
        # Search related code in memory
        from ..brain import search_all_memories
        memories = await search_all_memories(query, k=5)
        
        # Build response
        response_parts = [response]
        
        if memories:
            response_parts.append("\n\nRelated from memory:")
            for memory in memories[:3]:
                response_parts.append(f"- {memory.content[:100]}")
        
        return '\n'.join(response_parts)
    
    async def _get_context(self, file_path: str) -> Optional[EditorContext]:
        """Get editor context for file."""
        async with self._lock:
            return self._active_contexts.get(file_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get editor statistics."""
        return {
            'active_files': len(self._active_contexts),
            'files': list(self._active_contexts.keys())
        }


# Singleton
_editor: Optional[UnifiedAIEditor] = None
_editor_lock = asyncio.Lock()


async def get_unified_ai_editor() -> UnifiedAIEditor:
    """Get singleton unified AI editor."""
    global _editor
    if _editor is None:
        async with _editor_lock:
            if _editor is None:
                _editor = UnifiedAIEditor()
    return _editor


__all__ = [
    "UnifiedAIEditor",
    "EditorContext",
    "IntelligentEdit",
    "get_unified_ai_editor",
]
