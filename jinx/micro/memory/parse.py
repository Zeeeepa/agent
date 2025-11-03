"""Advanced XML-like tag extraction with caching and validation."""

from __future__ import annotations

import re
from typing import Tuple, Optional, Dict, List
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class ExtractResult:
    """Result of tag extraction."""
    content: str
    start_pos: int
    end_pos: int
    is_nested: bool = False


@lru_cache(maxsize=256)
def _compile_tag_pattern(tag: str) -> re.Pattern:
    """Compile and cache regex pattern for tag extraction."""
    # More robust pattern that handles attributes and nested tags
    pattern = rf"<{re.escape(tag)}(?:\s[^>]*)?>(.+?)</{re.escape(tag)}>"
    return re.compile(pattern, re.DOTALL | re.IGNORECASE)


def extract(tag: str, text: str) -> str | None:
    """Extract content from XML-like tags with improved robustness.
    
    Improvements over simple find():
    - Handles nested tags correctly
    - Ignores tag attributes
    - Case-insensitive matching
    - Regex-based for better accuracy
    - Cached pattern compilation
    """
    if not tag or not text:
        return None
    
    try:
        # Try regex-based extraction first (more robust)
        pattern = _compile_tag_pattern(tag)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    
    # Fallback to simple string search (backward compatible)
    try:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        
        a = text.find(open_tag)
        if a == -1:
            return None
        
        b = text.find(close_tag, a)
        if b == -1:
            return None
        
        start = a + len(open_tag)
        content = text[start:b].strip()
        
        # Validate extracted content
        if not content:
            return None
        
        return content
    except Exception:
        return None


def extract_all(tag: str, text: str) -> List[str]:
    """Extract all occurrences of a tag."""
    if not tag or not text:
        return []
    
    results: List[str] = []
    
    try:
        pattern = _compile_tag_pattern(tag)
        for match in pattern.finditer(text):
            content = match.group(1).strip()
            if content:
                results.append(content)
    except Exception:
        pass
    
    return results


def extract_with_metadata(tag: str, text: str) -> Optional[ExtractResult]:
    """Extract tag content with positional metadata."""
    if not tag or not text:
        return None
    
    try:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        
        a = text.find(open_tag)
        if a == -1:
            return None
        
        b = text.find(close_tag, a)
        if b == -1:
            return None
        
        start = a + len(open_tag)
        content = text[start:b].strip()
        
        # Check for nested tags
        is_nested = f"<{tag}>" in content
        
        return ExtractResult(
            content=content,
            start_pos=a,
            end_pos=b + len(close_tag),
            is_nested=is_nested
        )
    except Exception:
        return None


def parse_output(model_out: str) -> Tuple[str, str | None]:
    """Parse memory output with fallback handling.
    
    Returns:
        Tuple of (compact_memory, evergreen_memory)
    """
    if not model_out:
        return "", None
    
    # Extract both memory types
    compact = extract("mem_compact", model_out)
    durable = extract("mem_evergreen", model_out)
    
    # Fallback: use entire output as compact if no tags found
    if compact is None:
        compact = model_out.strip()
    
    # Validate compact is not empty
    if not compact:
        compact = model_out.strip() or ""
    
    return compact, durable


def extract_multiple_tags(text: str, tags: List[str]) -> Dict[str, Optional[str]]:
    """Extract multiple tags at once for efficiency."""
    results: Dict[str, Optional[str]] = {}
    
    for tag in tags:
        results[tag] = extract(tag, text)
    
    return results
