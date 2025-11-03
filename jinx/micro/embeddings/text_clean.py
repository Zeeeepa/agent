"""Advanced text cleaning with caching and robust pattern detection."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Set
from jinx.config import ALL_TAGS

# Intelligent patterns - structure-based, not keyword-based
_RE_NUMBERY = re.compile(r"^[\s\d+\-*/().]+$")  # Pure math expressions
_RE_WHITESPACE_ONLY = re.compile(r"^\s*$")
_RE_SPECIAL_CHARS = re.compile(r"^[<>{}\[\]()]+$")
_RE_STRUCTURAL_CODE = re.compile(r"[{};]|->|=>|\(.*\)\s*[:{]|\[.*\]\s*[=:]")
_RE_INDENT_PATTERN = re.compile(r"^\s{2,}\S")
_RE_OPERATOR_DENSE = re.compile(r"[=!<>+\-*/&|^%]{2,}|[=:]\s*[\[{]")


@lru_cache(maxsize=512)
def is_noise_text(pv: str) -> bool:
    """Detect noise text using intelligent structure analysis.
    
    Instead of hardcoded keywords, we analyze:
    - Text length and density
    - Character distribution
    - Structural patterns
    - Entropy and variety
    
    This approach is language-agnostic and ML-friendly.
    """
    if not pv:
        return True
    
    pv = pv.strip()
    
    # Too short
    if len(pv) < 4:
        return True
    
    # Whitespace only
    if _RE_WHITESPACE_ONLY.match(pv):
        return True
    
    # Special characters only
    if _RE_SPECIAL_CHARS.match(pv):
        return True
    
    # Number-only expressions (math formulas)
    if _RE_NUMBERY.match(pv) and any(ch.isdigit() for ch in pv):
        return True
    
    # Repeated single character (like "......" or "------")
    unique_chars = set(pv.replace(' ', '').replace('\t', ''))
    if len(unique_chars) == 1 and len(pv) > 3:
        return True
    
    # Very low entropy (too repetitive)
    if len(pv) > 10 and len(unique_chars) < 3:
        return True
    
    # Try ML-based detection if available
    try:
        from jinx.micro.text.heuristics import is_code_like
        # If detected as code by ML but too short, it's likely noise
        if is_code_like(pv) and len(pv) < 15:
            return True
    except Exception:
        pass
    
    return False


# Remove known wrapper tags like <machine_123>, </machine_123>, <python_...>
_TAG_OPEN_RE = re.compile(r"<([a-zA-Z_]+)(?:_\d+)?\s*>")
_TAG_CLOSE_RE = re.compile(r"</([a-zA-Z_]+)(?:_\d+)?\s*>")


# Cache for known tags set
_KNOWN_TAGS_SET: Set[str] = set()

def _get_known_tags() -> Set[str]:
    """Get known tags as a set (cached)."""
    global _KNOWN_TAGS_SET
    if not _KNOWN_TAGS_SET:
        try:
            _KNOWN_TAGS_SET = {tag.lower() for tag in ALL_TAGS}
        except Exception:
            _KNOWN_TAGS_SET = set()
    return _KNOWN_TAGS_SET


def strip_known_tags(text: str) -> str:
    """Strip known wrapper tags with intelligent filtering.
    
    Uses:
    - Cached tags set for O(1) lookup
    - Structural analysis instead of pattern matching
    - Entropy-based line filtering
    """
    if not text:
        return text
    
    known_tags = _get_known_tags()
    
    def repl_open(m: re.Match) -> str:
        base = m.group(1).lower()
        return "" if base in known_tags else m.group(0)
    
    def repl_close(m: re.Match) -> str:
        base = m.group(1).lower()
        return "" if base in known_tags else m.group(0)
    
    # Remove tags
    cleaned = _TAG_OPEN_RE.sub(repl_open, text)
    cleaned = _TAG_CLOSE_RE.sub(repl_close, cleaned)
    
    # Intelligent line filtering
    cleaned_lines = []
    
    for ln in cleaned.splitlines():
        s = ln.strip()
        
        # Skip empty lines
        if not s:
            continue
        
        # Use is_noise_text for intelligent filtering
        if is_noise_text(s):
            continue
        
        # Skip lines with only punctuation/brackets (low entropy)
        if len(s) > 0 and all(not ch.isalnum() and not ch.isspace() for ch in s):
            continue
        
        cleaned_lines.append(ln)
    
    return "\n".join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace intelligently."""
    if not text:
        return text
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace from lines
    lines = [ln.rstrip() for ln in text.splitlines()]
    
    return '\n'.join(lines)


def analyze_text_structure(text: str) -> dict:
    """Analyze text structure for intelligent processing.
    
    Returns metrics instead of boolean flags:
    - code_density: 0.0-1.0
    - operator_ratio: 0.0-1.0  
    - indent_pattern: bool
    - entropy: float
    - symbol_ratio: 0.0-1.0
    """
    if not text:
        return {
            "code_density": 0.0,
            "operator_ratio": 0.0,
            "indent_pattern": False,
            "entropy": 0.0,
            "symbol_ratio": 0.0
        }
    
    text = text.strip()
    length = len(text)
    
    # Count structural patterns (not keywords!)
    structural_matches = len(_RE_STRUCTURAL_CODE.findall(text))
    code_density = min(1.0, structural_matches / max(1, length / 10))
    
    # Operator density
    operator_matches = len(_RE_OPERATOR_DENSE.findall(text))
    operator_ratio = operator_matches / max(1, len(text.split()))
    
    # Indent pattern detection
    lines = text.splitlines()
    indent_pattern = sum(1 for ln in lines if _RE_INDENT_PATTERN.match(ln)) > len(lines) * 0.3
    
    # Character entropy (Shannon-like)
    if length > 0:
        char_freq = {}
        for ch in text:
            char_freq[ch] = char_freq.get(ch, 0) + 1
        
        import math
        entropy = -sum((freq/length) * math.log2(freq/length) for freq in char_freq.values())
    else:
        entropy = 0.0
    
    # Symbol ratio (structural characters vs alphanumeric)
    symbols = sum(1 for ch in text if ch in '{}[]()<>:;,=+-*/%&|^!~')
    symbol_ratio = symbols / max(1, length)
    
    return {
        "code_density": code_density,
        "operator_ratio": operator_ratio,
        "indent_pattern": indent_pattern,
        "entropy": entropy,
        "symbol_ratio": symbol_ratio
    }
