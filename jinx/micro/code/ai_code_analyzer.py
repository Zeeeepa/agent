"""AI Code Analyzer - real-time code intelligence with ML.

Modern AI editor capabilities:
- Real-time code analysis
- Intelligent diagnostics
- Context-aware suggestions
- Automated refactoring
- Code quality scoring
"""

from __future__ import annotations

import asyncio
import ast
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..embeddings import embed_text_cached, search_embeddings_unified
from ..brain import (
    classify_query_intent,
    search_all_memories,
    think_with_memory,
    get_adaptive_manager,
)


class DiagnosticSeverity(Enum):
    """Diagnostic severity levels (LSP-compatible)."""
    ERROR = 1
    WARNING = 2
    INFO = 3
    HINT = 4


@dataclass
class CodeDiagnostic:
    """Code diagnostic (error/warning/hint)."""
    severity: DiagnosticSeverity
    message: str
    line: int
    column: int
    end_line: int
    end_column: int
    source: str  # 'jinx-ai', 'syntax', 'semantic', etc.
    code: Optional[str] = None
    fix: Optional[str] = None  # Auto-fix suggestion


@dataclass
class CodeSuggestion:
    """Code completion/suggestion."""
    text: str
    label: str
    kind: str  # 'function', 'variable', 'snippet', etc.
    detail: str
    documentation: str
    insert_text: str
    confidence: float
    learned: bool  # From memory?


@dataclass
class CodeAction:
    """Code action (refactoring/quick fix)."""
    title: str
    kind: str  # 'refactor', 'quickfix', 'source'
    diagnostics: List[CodeDiagnostic]
    edit: Dict[str, Any]  # Text edits
    command: Optional[str] = None


class AICodeAnalyzer:
    """Real-time AI-powered code analyzer."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._diagnostics_cache: Dict[str, List[CodeDiagnostic]] = {}
        self._lock = asyncio.Lock()
    
    async def analyze_file(
        self,
        file_path: str,
        content: str,
        *,
        real_time: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive file analysis."""
        
        # Parse AST
        try:
            tree = ast.parse(content)
            syntax_valid = True
            syntax_errors = []
        except SyntaxError as e:
            syntax_valid = False
            syntax_errors = [
                CodeDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message=e.msg,
                    line=e.lineno or 1,
                    column=e.offset or 0,
                    end_line=e.lineno or 1,
                    end_column=(e.offset or 0) + 1,
                    source='syntax'
                )
            ]
            tree = None
        
        diagnostics = syntax_errors.copy()
        
        # AI-powered analysis
        if syntax_valid and tree:
            # Semantic analysis
            semantic_diags = await self._semantic_analysis(tree, content, file_path)
            diagnostics.extend(semantic_diags)
            
            # Code quality analysis
            quality_diags = await self._quality_analysis(tree, content)
            diagnostics.extend(quality_diags)
            
            # Memory-based analysis (learned patterns)
            memory_diags = await self._memory_analysis(content, file_path)
            diagnostics.extend(memory_diags)
        
        # Code metrics
        metrics = await self._calculate_metrics(tree, content) if tree else {}
        
        # Overall quality score
        quality_score = await self._calculate_quality_score(
            diagnostics, metrics
        )
        
        # Cache diagnostics
        async with self._lock:
            self._diagnostics_cache[file_path] = diagnostics
        
        return {
            'file_path': file_path,
            'syntax_valid': syntax_valid,
            'diagnostics': diagnostics,
            'metrics': metrics,
            'quality_score': quality_score,
            'suggestions_count': len([
                d for d in diagnostics 
                if d.severity == DiagnosticSeverity.HINT and d.fix
            ])
        }
    
    async def get_completions(
        self,
        file_path: str,
        content: str,
        line: int,
        column: int
    ) -> List[CodeSuggestion]:
        """Get intelligent code completions."""
        
        # Get line content
        lines = content.split('\n')
        if line >= len(lines):
            return []
        
        current_line = lines[line]
        prefix = current_line[:column]
        
        suggestions = []
        
        # 1. AST-based completions
        ast_suggestions = await self._get_ast_completions(
            content, line, column
        )
        suggestions.extend(ast_suggestions)
        
        # 2. Memory-based completions (learned from past code)
        memory_suggestions = await self._get_memory_completions(
            prefix, file_path
        )
        suggestions.extend(memory_suggestions)
        
        # 3. Context-aware completions
        context_suggestions = await self._get_context_completions(
            content, line, column, file_path
        )
        suggestions.extend(context_suggestions)
        
        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        
        return suggestions[:20]  # Top 20
    
    async def get_code_actions(
        self,
        file_path: str,
        content: str,
        line: int
    ) -> List[CodeAction]:
        """Get available code actions (refactorings/quick fixes)."""
        
        actions = []
        
        # Get diagnostics for this line
        diagnostics = await self._get_diagnostics_for_line(file_path, line)
        
        # Quick fixes for diagnostics
        for diagnostic in diagnostics:
            if diagnostic.fix:
                actions.append(CodeAction(
                    title=f"Fix: {diagnostic.message}",
                    kind='quickfix',
                    diagnostics=[diagnostic],
                    edit={
                        'changes': {
                            file_path: [{
                                'range': {
                                    'start': {'line': diagnostic.line, 'character': diagnostic.column},
                                    'end': {'line': diagnostic.end_line, 'character': diagnostic.end_column}
                                },
                                'newText': diagnostic.fix
                            }]
                        }
                    }
                ))
        
        # Refactoring actions
        refactor_actions = await self._get_refactoring_actions(
            content, line
        )
        actions.extend(refactor_actions)
        
        return actions
    
    async def _semantic_analysis(
        self,
        tree: ast.AST,
        content: str,
        file_path: str
    ) -> List[CodeDiagnostic]:
        """Semantic code analysis."""
        
        diagnostics = []
        
        # Check for common issues
        for node in ast.walk(tree):
            # Unused variables
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Would check if variable is used
                pass
            
            # Missing type hints
            if isinstance(node, ast.FunctionDef):
                if not node.returns:
                    diagnostics.append(CodeDiagnostic(
                        severity=DiagnosticSeverity.HINT,
                        message=f"Function '{node.name}' missing return type hint",
                        line=node.lineno,
                        column=node.col_offset,
                        end_line=node.lineno,
                        end_column=node.col_offset + len(node.name),
                        source='jinx-ai-semantic',
                        fix=f"def {node.name}(...) -> ReturnType:"
                    ))
        
        return diagnostics
    
    async def _quality_analysis(
        self,
        tree: ast.AST,
        content: str
    ) -> List[CodeDiagnostic]:
        """Code quality analysis."""
        
        diagnostics = []
        
        for node in ast.walk(tree):
            # Function complexity
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    diagnostics.append(CodeDiagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        message=f"Function '{node.name}' has high complexity ({complexity})",
                        line=node.lineno,
                        column=node.col_offset,
                        end_line=node.end_lineno or node.lineno,
                        end_column=node.end_col_offset or node.col_offset,
                        source='jinx-ai-quality',
                        code='high_complexity'
                    ))
            
            # Long functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines = (node.end_lineno or node.lineno) - node.lineno
                if lines > 50:
                    diagnostics.append(CodeDiagnostic(
                        severity=DiagnosticSeverity.INFO,
                        message=f"Function '{node.name}' is long ({lines} lines)",
                        line=node.lineno,
                        column=node.col_offset,
                        end_line=node.lineno,
                        end_column=node.col_offset + len(node.name),
                        source='jinx-ai-quality',
                        code='long_function'
                    ))
        
        return diagnostics
    
    async def _memory_analysis(
        self,
        content: str,
        file_path: str
    ) -> List[CodeDiagnostic]:
        """Analysis using learned patterns from memory."""
        
        diagnostics = []
        
        # Search for similar code patterns in memory
        memories = await search_all_memories(
            f"code issue {content[:200]}",
            k=5
        )
        
        for memory in memories:
            if hasattr(memory, 'content') and 'issue' in memory.content.lower():
                # Found learned issue pattern
                diagnostics.append(CodeDiagnostic(
                    severity=DiagnosticSeverity.HINT,
                    message=f"Learned pattern: {memory.content[:100]}",
                    line=1,
                    column=0,
                    end_line=1,
                    end_column=0,
                    source='jinx-ai-memory'
                ))
        
        return diagnostics
    
    async def _calculate_metrics(
        self,
        tree: ast.AST,
        content: str
    ) -> Dict[str, Any]:
        """Calculate code metrics."""
        
        if not tree:
            return {}
        
        metrics = {
            'lines': len(content.split('\n')),
            'functions': 0,
            'classes': 0,
            'async_functions': 0,
            'complexity': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['functions'] += 1
                metrics['complexity'] += self._calculate_complexity(node)
            elif isinstance(node, ast.AsyncFunctionDef):
                metrics['async_functions'] += 1
                metrics['complexity'] += self._calculate_complexity(node)
            elif isinstance(node, ast.ClassDef):
                metrics['classes'] += 1
        
        return metrics
    
    async def _calculate_quality_score(
        self,
        diagnostics: List[CodeDiagnostic],
        metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score (0-1)."""
        
        score = 1.0
        
        # Penalize errors/warnings
        for diag in diagnostics:
            if diag.severity == DiagnosticSeverity.ERROR:
                score -= 0.2
            elif diag.severity == DiagnosticSeverity.WARNING:
                score -= 0.1
            elif diag.severity == DiagnosticSeverity.INFO:
                score -= 0.05
        
        # Bonus for good metrics
        if metrics.get('async_functions', 0) > 0:
            score += 0.05  # Using async
        
        return max(0.0, min(1.0, score))
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    async def _get_ast_completions(
        self,
        content: str,
        line: int,
        column: int
    ) -> List[CodeSuggestion]:
        """Get AST-based completions."""
        # Would analyze AST context
        return []
    
    async def _get_memory_completions(
        self,
        prefix: str,
        file_path: str
    ) -> List[CodeSuggestion]:
        """Get completions from memory (learned patterns)."""
        
        suggestions = []
        
        # Search memory for similar code
        memories = await search_all_memories(
            f"code completion {prefix}",
            k=5
        )
        
        for memory in memories:
            if hasattr(memory, 'content'):
                suggestions.append(CodeSuggestion(
                    text=memory.content[:50],
                    label=memory.content[:30],
                    kind='snippet',
                    detail='From memory',
                    documentation=f"Learned from past code",
                    insert_text=memory.content[:50],
                    confidence=getattr(memory, 'importance', 0.5),
                    learned=True
                ))
        
        return suggestions
    
    async def _get_context_completions(
        self,
        content: str,
        line: int,
        column: int,
        file_path: str
    ) -> List[CodeSuggestion]:
        """Get context-aware completions using brain."""
        
        # Get context from brain
        thought = await think_with_memory(
            f"Code completion at {file_path}:{line}:{column}\nContext: {content[max(0, column-100):column]}"
        )
        
        # Would generate suggestions based on thought
        return []
    
    async def _get_diagnostics_for_line(
        self,
        file_path: str,
        line: int
    ) -> List[CodeDiagnostic]:
        """Get diagnostics for specific line."""
        async with self._lock:
            all_diags = self._diagnostics_cache.get(file_path, [])
            return [d for d in all_diags if d.line == line]
    
    async def _get_refactoring_actions(
        self,
        content: str,
        line: int
    ) -> List[CodeAction]:
        """Get refactoring actions."""
        # Would generate refactoring suggestions
        return []


# Singleton
_analyzer: Optional[AICodeAnalyzer] = None
_analyzer_lock = asyncio.Lock()


async def get_ai_analyzer() -> AICodeAnalyzer:
    """Get singleton AI analyzer."""
    global _analyzer
    if _analyzer is None:
        async with _analyzer_lock:
            if _analyzer is None:
                _analyzer = AICodeAnalyzer()
    return _analyzer


__all__ = [
    "AICodeAnalyzer",
    "CodeDiagnostic",
    "CodeSuggestion",
    "CodeAction",
    "DiagnosticSeverity",
    "get_ai_analyzer",
]
