"""Semantic Patcher - ML-powered code patching with embeddings and brain integration.

Uses embeddings for semantic code search and pattern learning from patch history.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..embeddings.embed_cache import embed_text_cached
from ..brain import (
    search_all_memories,
    remember_episode,
    get_knowledge_graph,
    think_with_memory,
)


@dataclass
class CodePattern:
    """Learned code pattern from past patches."""
    pattern_id: str
    code_snippet: str
    embedding: List[float]
    patch_count: int
    success_rate: float
    common_transforms: List[str]
    context: Dict[str, Any]


@dataclass
class SemanticMatch:
    """Semantic code match result."""
    original_code: str
    similar_pattern: CodePattern
    similarity_score: float
    suggested_patch: Optional[str]
    confidence: float


class SemanticPatcher:
    """ML-powered semantic code patcher."""
    
    def __init__(self):
        self._pattern_cache: Dict[str, CodePattern] = {}
        self._lock = asyncio.Lock()
    
    async def find_similar_code(
        self,
        code_snippet: str,
        *,
        min_similarity: float = 0.75
    ) -> List[SemanticMatch]:
        """Find semantically similar code using embeddings."""
        
        # Embed the query code
        query_embedding = await embed_text_cached(code_snippet[:512])
        
        if not query_embedding:
            return []
        
        # Search memories for similar code patterns
        memories = await search_all_memories(code_snippet, k=10)
        
        matches = []
        
        for memory in memories:
            # Check if memory contains code
            if 'code' not in memory.content.lower():
                continue
            
            # Compute similarity
            if memory.embedding:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
                
                if similarity >= min_similarity:
                    # Extract pattern from memory
                    pattern = await self._extract_pattern_from_memory(memory)
                    
                    if pattern:
                        # Get suggested patch from pattern
                        suggested_patch = await self._suggest_patch_from_pattern(
                            code_snippet,
                            pattern
                        )
                        
                        matches.append(SemanticMatch(
                            original_code=code_snippet,
                            similar_pattern=pattern,
                            similarity_score=similarity,
                            suggested_patch=suggested_patch,
                            confidence=similarity * pattern.success_rate
                        ))
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return matches
    
    async def learn_from_patch(
        self,
        original_code: str,
        patched_code: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Learn from patch application for future suggestions."""
        
        # Create pattern hash
        pattern_hash = hashlib.sha256(original_code.encode()).hexdigest()[:16]
        
        async with self._lock:
            if pattern_hash in self._pattern_cache:
                # Update existing pattern
                pattern = self._pattern_cache[pattern_hash]
                pattern.patch_count += 1
                
                # Update success rate
                old_total = pattern.patch_count - 1
                pattern.success_rate = (
                    (pattern.success_rate * old_total + (1.0 if success else 0.0))
                    / pattern.patch_count
                )
                
                # Update transforms
                transform = self._extract_transform(original_code, patched_code)
                if transform and transform not in pattern.common_transforms:
                    pattern.common_transforms.append(transform)
            else:
                # Create new pattern
                embedding = await embed_text_cached(original_code[:512])
                
                pattern = CodePattern(
                    pattern_id=pattern_hash,
                    code_snippet=original_code,
                    embedding=embedding or [],
                    patch_count=1,
                    success_rate=1.0 if success else 0.0,
                    common_transforms=[self._extract_transform(original_code, patched_code)],
                    context=context or {}
                )
                
                self._pattern_cache[pattern_hash] = pattern
        
        # Store in brain memory
        await remember_episode(
            content=f"Code patch: {original_code[:100]}... -> {patched_code[:100]}...",
            episode_type='tool_use',
            context={
                'pattern_id': pattern_hash,
                'success': success,
                'transform': self._extract_transform(original_code, patched_code),
                **(context or {})
            },
            importance=0.8 if success else 0.6
        )
    
    async def suggest_patches_for_file(
        self,
        file_path: str,
        code_content: str
    ) -> List[Dict[str, Any]]:
        """Suggest patches for entire file using learned patterns."""
        
        suggestions = []
        
        # Split into functions/classes
        code_blocks = await self._extract_code_blocks(code_content)
        
        for block in code_blocks:
            # Find similar patterns
            matches = await self.find_similar_code(
                block['code'],
                min_similarity=0.75
            )
            
            for match in matches:
                if match.suggested_patch and match.confidence >= 0.7:
                    suggestions.append({
                        'location': block['location'],
                        'original': block['code'],
                        'suggested': match.suggested_patch,
                        'confidence': match.confidence,
                        'reason': f"Based on {match.similar_pattern.patch_count} similar patches"
                    })
        
        return suggestions
    
    async def intelligent_refactor(
        self,
        code: str,
        intent: str
    ) -> Optional[str]:
        """Refactor code based on natural language intent using brain."""
        
        # Use brain's thinking to understand intent
        thought = await think_with_memory(
            f"Refactor this code: {code[:200]}...\nIntent: {intent}"
        )
        
        # Search for similar refactorings in memory
        memories = await search_all_memories(
            f"refactor {intent}",
            k=5
        )
        
        # Extract patterns from memories
        patterns = []
        for memory in memories:
            if 'before' in memory.content and 'after' in memory.content:
                patterns.append(memory)
        
        if not patterns:
            return None
        
        # Apply most relevant pattern
        # This is simplified - in production would use LLM
        # For now, return the thought as guidance
        return f"# Refactoring suggestion based on brain intelligence:\n# {thought}\n\n{code}"
    
    async def detect_code_smells(
        self,
        code: str
    ) -> List[Dict[str, Any]]:
        """Detect code smells using learned patterns."""
        
        smells = []
        
        # Check for common anti-patterns learned from past patches
        antipatterns = [
            {
                'pattern': 'except Exception:\n    pass',
                'smell': 'Silent exception handling',
                'severity': 'high'
            },
            {
                'pattern': 'while True:\n',
                'smell': 'Infinite loop without break',
                'severity': 'medium'
            },
            {
                'pattern': 'global ',
                'smell': 'Global variable usage',
                'severity': 'medium'
            }
        ]
        
        for antipattern in antipatterns:
            if antipattern['pattern'] in code:
                smells.append({
                    'smell': antipattern['smell'],
                    'severity': antipattern['severity'],
                    'location': code.find(antipattern['pattern'])
                })
        
        # Use brain to find learned smells
        kg = await get_knowledge_graph()
        smell_patterns = await kg.query_patterns("code smell", "similar")
        
        for pattern in smell_patterns[:5]:
            pattern_code = pattern.get('data', {}).get('pattern', '')
            if pattern_code and pattern_code in code:
                smells.append({
                    'smell': pattern.get('data', {}).get('description', 'Unknown'),
                    'severity': 'low',
                    'location': code.find(pattern_code),
                    'learned': True
                })
        
        return smells
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    async def _extract_pattern_from_memory(
        self,
        memory: Any
    ) -> Optional[CodePattern]:
        """Extract code pattern from memory."""
        # Try to parse memory content for code pattern
        content = memory.content
        
        # Look for code markers
        if '```' in content:
            # Extract code from markdown
            parts = content.split('```')
            if len(parts) >= 2:
                code = parts[1].split('\n', 1)[1] if '\n' in parts[1] else parts[1]
                
                return CodePattern(
                    pattern_id=memory.id if hasattr(memory, 'id') else 'unknown',
                    code_snippet=code,
                    embedding=memory.embedding if hasattr(memory, 'embedding') else [],
                    patch_count=1,
                    success_rate=memory.importance if hasattr(memory, 'importance') else 0.5,
                    common_transforms=[],
                    context={}
                )
        
        return None
    
    async def _suggest_patch_from_pattern(
        self,
        code: str,
        pattern: CodePattern
    ) -> Optional[str]:
        """Suggest patch based on learned pattern."""
        # Apply common transforms from pattern
        suggested = code
        
        for transform in pattern.common_transforms:
            # Simple transform application
            if ' -> ' in transform:
                old, new = transform.split(' -> ', 1)
                suggested = suggested.replace(old.strip(), new.strip())
        
        return suggested if suggested != code else None
    
    def _extract_transform(self, original: str, patched: str) -> str:
        """Extract transformation description."""
        # Simple diff-like transform
        if len(original) < 100 and len(patched) < 100:
            return f"{original.strip()} -> {patched.strip()}"
        return "complex_transform"
    
    async def _extract_code_blocks(
        self,
        code: str
    ) -> List[Dict[str, Any]]:
        """Extract code blocks (functions, classes) from code."""
        import ast
        
        blocks = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = node.lineno
                    end = node.end_lineno or start
                    
                    lines = code.split('\n')
                    block_code = '\n'.join(lines[start-1:end])
                    
                    blocks.append({
                        'type': type(node).__name__,
                        'name': node.name,
                        'location': f"line {start}-{end}",
                        'code': block_code
                    })
        except SyntaxError:
            # If can't parse, return whole file as one block
            blocks.append({
                'type': 'file',
                'name': 'entire_file',
                'location': 'full',
                'code': code
            })
        
        return blocks
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        total_patterns = len(self._pattern_cache)
        
        if not self._pattern_cache:
            return {
                'total_patterns': 0,
                'avg_success_rate': 0.0,
                'total_patches': 0
            }
        
        avg_success = sum(p.success_rate for p in self._pattern_cache.values()) / total_patterns
        total_patches = sum(p.patch_count for p in self._pattern_cache.values())
        
        return {
            'total_patterns': total_patterns,
            'avg_success_rate': avg_success,
            'total_patches': total_patches,
            'patterns': [
                {
                    'id': p.pattern_id,
                    'patches': p.patch_count,
                    'success_rate': p.success_rate
                }
                for p in sorted(
                    self._pattern_cache.values(),
                    key=lambda x: x.patch_count,
                    reverse=True
                )[:10]
            ]
        }


# Singleton
_semantic_patcher: Optional[SemanticPatcher] = None
_patcher_lock = asyncio.Lock()


async def get_semantic_patcher() -> SemanticPatcher:
    """Get singleton semantic patcher."""
    global _semantic_patcher
    if _semantic_patcher is None:
        async with _patcher_lock:
            if _semantic_patcher is None:
                _semantic_patcher = SemanticPatcher()
    return _semantic_patcher


async def find_semantic_matches(code: str, **kwargs) -> List[SemanticMatch]:
    """Find semantically similar code patterns."""
    patcher = await get_semantic_patcher()
    return await patcher.find_similar_code(code, **kwargs)


async def suggest_intelligent_patches(file_path: str, code: str) -> List[Dict[str, Any]]:
    """Suggest patches using ML and brain intelligence."""
    patcher = await get_semantic_patcher()
    return await patcher.suggest_patches_for_file(file_path, code)


__all__ = [
    "SemanticPatcher",
    "CodePattern",
    "SemanticMatch",
    "get_semantic_patcher",
    "find_semantic_matches",
    "suggest_intelligent_patches",
]
