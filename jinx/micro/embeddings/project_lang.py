"""Intelligent language detection using structure and content analysis."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional
import os


# Extension mappings (used as fallback)
_EXT_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.jsx': 'jsx',
    '.go': 'go',
    '.java': 'java',
    '.cs': 'csharp',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.c': 'c',
    '.rs': 'rust',
    '.php': 'php',
    '.rb': 'ruby',
    '.sh': 'bash',
    '.bash': 'bash',
    '.ps1': 'powershell',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.ini': 'ini',
    '.md': 'markdown',
}


@lru_cache(maxsize=512)
def lang_for_file(path: str) -> str:
    """Detect language intelligently using extension mapping.
    
    This is a fast path using extension mapping.
    For content-based detection, use lang_for_content().
    """
    if not path:
        return ''
    
    # Get extension
    _, ext = os.path.splitext(path.lower())
    
    # Lookup in map
    return _EXT_MAP.get(ext, '')


def lang_for_content(content: str, hint_path: Optional[str] = None) -> str:
    """Detect language from content using structural analysis.
    
    This doesn't rely on hardcoded keywords but analyzes:
    - Syntax patterns
    - Structural features
    - Character distributions
    """
    if not content:
        return ''
    
    # Try extension hint first
    if hint_path:
        ext_lang = lang_for_file(hint_path)
        if ext_lang:
            return ext_lang
    
    # Analyze content structure
    content = content.strip()
    
    # Structural analysis (no keywords!)
    has_braces = '{' in content and '}' in content
    has_semicolons = ';' in content
    has_colon_indent = ':' in content and '\n    ' in content
    has_arrow_funcs = '=>' in content
    has_angle_brackets = '<' in content and '>' in content
    
    # Indentation-based (Python-like)
    if has_colon_indent and not has_braces and not has_semicolons:
        return 'python'
    
    # C-family (braces + semicolons)
    if has_braces and has_semicolons:
        if has_arrow_funcs:
            return 'javascript'  # or typescript
        if has_angle_brackets:
            return 'cpp'  # or java
        return 'c'
    
    # Fallback to empty
    return ''
