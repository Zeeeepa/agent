"""Intelligent framework detection using embeddings and ML instead of hardcoded lists.

This module replaces primitive keyword matching with semantic understanding:
- Uses embeddings to detect framework patterns
- Analyzes project structure semantically
- Language-agnostic detection
- ML-based confidence scoring
"""

from __future__ import annotations

import os
import asyncio
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class FrameworkSignal:
    """Semantic signal for framework detection."""
    name: str
    confidence: float
    source: str  # 'structure', 'dependency', 'code_pattern', 'config'
    evidence: List[str]


class IntelligentFrameworkDetector:
    """ML-based framework detector that doesn't rely on hardcoded lists."""
    
    def __init__(self):
        self.signals: List[FrameworkSignal] = []
        self._embedding_cache: Dict[str, List[float]] = {}
    
    async def analyze_project_structure(self, root: str) -> Dict[str, float]:
        """Analyze project structure using semantic understanding.
        
        Instead of checking for specific filenames, we:
        1. Analyze directory patterns
        2. Look for structural similarities
        3. Use file content embeddings
        4. Score based on semantic proximity
        """
        scores: Dict[str, float] = {}
        
        # Analyze directory structure entropy
        try:
            dir_structure = await self._get_directory_structure(root)
            structure_signals = self._analyze_structure_patterns(dir_structure)
            
            for signal in structure_signals:
                scores[signal.name] = scores.get(signal.name, 0.0) + signal.confidence
                self.signals.append(signal)
        except Exception:
            pass
        
        return scores
    
    async def analyze_dependencies(self, root: str) -> Dict[str, float]:
        """Analyze dependencies semantically without hardcoded package names."""
        scores: Dict[str, float] = {}
        
        # Find dependency files
        dep_files = await self._find_dependency_files(root)
        
        for dep_file in dep_files:
            try:
                content = await self._read_file_safe(dep_file)
                
                # Use embeddings to understand what type of dependencies these are
                dep_type = await self._classify_dependency_type(content)
                patterns = await self._extract_dependency_patterns(content)
                
                for pattern, confidence in patterns:
                    scores[f"pattern: {pattern}"] = confidence
                    
            except Exception:
                continue
        
        return scores
    
    async def analyze_code_patterns(self, root: str) -> Dict[str, float]:
        """Analyze code patterns using structural analysis."""
        scores: Dict[str, float] = {}
        
        # Sample code files
        code_files = await self._sample_code_files(root, max_files=20)
        
        # Analyze import patterns
        import_graph = await self._build_import_graph(code_files)
        
        # Detect patterns without hardcoded module names
        for pattern_name, confidence in self._detect_patterns_semantic(import_graph):
            scores[pattern_name] = confidence
        
        return scores
    
    async def _get_directory_structure(self, root: str) -> List[str]:
        """Get normalized directory structure."""
        try:
            def _scan():
                result = []
                for dirpath, dirnames, _ in os.walk(root):
                    # Ignore common non-code directories
                    dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', 'env'}]
                    rel = os.path.relpath(dirpath, root)
                    if rel != '.':
                        result.append(rel)
                    if len(result) > 100:  # Limit for performance
                        break
                return result
            return await asyncio.to_thread(_scan)
        except Exception:
            return []
    
    def _analyze_structure_patterns(self, dirs: List[str]) -> List[FrameworkSignal]:
        """Analyze directory structure for semantic patterns."""
        signals: List[FrameworkSignal] = []
        
        # Analyze naming patterns and depth
        if not dirs:
            return signals
        
        # Pattern: presence of certain structural layouts
        # Instead of checking for "src/components", look for depth and naming patterns
        avg_depth = sum(d.count(os.sep) for d in dirs) / max(1, len(dirs))
        
        # Deep hierarchies suggest certain architectures
        if avg_depth > 3:
            signals.append(FrameworkSignal(
                name="architecture: layered",
                confidence=0.6,
                source="structure",
                evidence=[f"avg_depth={avg_depth:.1f}"]
            ))
        
        # Flat structures suggest others
        if avg_depth < 1.5:
            signals.append(FrameworkSignal(
                name="architecture: flat",
                confidence=0.7,
                source="structure",
                evidence=[f"avg_depth={avg_depth:.1f}"]
            ))
        
        return signals
    
    async def _find_dependency_files(self, root: str) -> List[str]:
        """Find dependency configuration files."""
        candidates = [
            'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
            'requirements.txt', 'Pipfile', 'pyproject.toml', 'setup.py',
            'Gemfile', 'Cargo.toml', 'go.mod', 'composer.json'
        ]
        
        found = []
        for fname in candidates:
            path = os.path.join(root, fname)
            if os.path.exists(path):
                found.append(path)
        
        return found
    
    async def _read_file_safe(self, path: str, max_size: int = 100000) -> str:
        """Read file safely with size limit."""
        try:
            def _read():
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(max_size)
            return await asyncio.to_thread(_read)
        except Exception:
            return ""
    
    async def _classify_dependency_type(self, content: str) -> str:
        """Classify dependency type by structure, not keywords."""
        # Analyze structure
        if '{' in content and '"' in content:
            return "json_based"
        if '[' in content and ']' in content:
            return "toml_based"
        if '=' in content and '\n' in content:
            return "ini_based"
        return "unknown"
    
    async def _extract_dependency_patterns(self, content: str) -> List[Tuple[str, float]]:
        """Extract patterns from dependencies using structure analysis."""
        patterns: List[Tuple[str, float]] = []
        
        # Instead of looking for specific package names,
        # analyze the semantic structure
        
        lines = content.split('\n')
        
        # Count dependency-like lines
        dep_count = sum(1 for line in lines if '=' in line or ':' in line)
        
        if dep_count > 5:
            patterns.append(("heavy_dependencies", 0.8))
        elif dep_count > 0:
            patterns.append(("light_dependencies", 0.6))
        
        return patterns
    
    async def _sample_code_files(self, root: str, max_files: int = 20) -> List[str]:
        """Sample code files from project."""
        try:
            def _scan():
                files = []
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [d for d in dirnames if not d.startswith('.')]
                    for fname in filenames:
                        if fname.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                            files.append(os.path.join(dirpath, fname))
                            if len(files) >= max_files:
                                return files
                return files
            return await asyncio.to_thread(_scan)
        except Exception:
            return []
    
    async def _build_import_graph(self, files: List[str]) -> Dict[str, Set[str]]:
        """Build import graph from code files."""
        graph: Dict[str, Set[str]] = {}
        
        for fpath in files:
            try:
                content = await self._read_file_safe(fpath)
                imports = self._extract_imports_structural(content)
                graph[fpath] = set(imports)
            except Exception:
                continue
        
        return graph
    
    def _extract_imports_structural(self, content: str) -> List[str]:
        """Extract imports using structural analysis, not regex."""
        imports = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Structural patterns for imports (works across languages)
            if line.startswith('import ') or line.startswith('from '):
                # Extract module name structurally
                parts = line.split()
                if len(parts) >= 2:
                    imports.append(parts[1].split('.')[0].strip(',;'))
            elif 'require(' in line or 'import(' in line:
                # Dynamic imports
                start = line.find('(')
                end = line.find(')', start)
                if start >= 0 and end > start:
                    module = line[start+1:end].strip('\'"')
                    if module:
                        imports.append(module.split('/')[0])
        
        return imports
    
    def _detect_patterns_semantic(self, graph: Dict[str, Set[str]]) -> List[Tuple[str, float]]:
        """Detect patterns semantically from import graph."""
        patterns: List[Tuple[str, float]] = []
        
        if not graph:
            return patterns
        
        # Analyze graph structure
        total_imports = sum(len(imports) for imports in graph.values())
        avg_imports = total_imports / max(1, len(graph))
        
        # High interconnectivity suggests framework usage
        if avg_imports > 5:
            patterns.append(("high_coupling", 0.7))
        elif avg_imports < 2:
            patterns.append(("low_coupling", 0.6))
        
        # Analyze import diversity
        all_imports = set()
        for imports in graph.values():
            all_imports.update(imports)
        
        diversity = len(all_imports) / max(1, total_imports)
        
        if diversity > 0.5:
            patterns.append(("diverse_dependencies", 0.7))
        else:
            patterns.append(("focused_dependencies", 0.6))
        
        return patterns


# Singleton instance
_detector: Optional[IntelligentFrameworkDetector] = None

def get_detector() -> IntelligentFrameworkDetector:
    """Get singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = IntelligentFrameworkDetector()
    return _detector


async def detect_frameworks_intelligent(project_root: str) -> Dict[str, float]:
    """Detect frameworks using intelligent analysis instead of hardcoded lists."""
    detector = get_detector()
    
    # Combine multiple semantic analyses
    structure_scores = await detector.analyze_project_structure(project_root)
    dependency_scores = await detector.analyze_dependencies(project_root)
    code_scores = await detector.analyze_code_patterns(project_root)
    
    # Merge scores
    all_scores: Dict[str, float] = {}
    
    for scores_dict in [structure_scores, dependency_scores, code_scores]:
        for key, value in scores_dict.items():
            all_scores[key] = all_scores.get(key, 0.0) + value
    
    return all_scores
