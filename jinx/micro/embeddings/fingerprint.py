"""Optimized simhash fingerprinting with caching and performance improvements."""

from __future__ import annotations

import hashlib
from typing import Iterable, List
from functools import lru_cache

# Simple simhash-like fingerprint for text chunks to support dedup/consolidation.
# Not cryptographic. Produces a fixed-length hex string (64-bit by default).

_DEF_BITS = 64

# Pre-computed bit masks for performance
_BIT_MASKS = [1 << i for i in range(_DEF_BITS)]


def _tokens(text: str) -> List[str]:
    """Tokenize text efficiently.
    
    Optimizations:
    - Single pass through text
    - List-based accumulation
    - Minimal string operations
    """
    if not text:
        return []
    
    t = text.lower()
    tokens: List[str] = []
    cur: List[str] = []
    
    for ch in t:
        if ch.isalnum() or ch == '_':
            cur.append(ch)
        else:
            if cur:
                token = "".join(cur)
                if len(token) >= 2:  # Filter very short tokens
                    tokens.append(token)
                cur = []
    
    # Don't forget last token
    if cur:
        token = "".join(cur)
        if len(token) >= 2:
            tokens.append(token)
    
    return tokens


@lru_cache(maxsize=512)
def simhash(text: str, *, bits: int = _DEF_BITS) -> str:
    """Compute simhash fingerprint with optimizations.
    
    Improvements:
    - Cached for repeated texts
    - Pre-computed bit masks
    - Optimized hash computation
    - Better weight calculation
    
    Args:
        text: Input text to fingerprint
        bits: Number of bits in hash (default: 64)
    
    Returns:
        Hex string representation of hash
    """
    if not text:
        return "0" * (bits // 4)
    
    # Initialize vector
    v = [0] * bits
    tokens = _tokens(text)
    
    if not tokens:
        return "0" * (bits // 4)
    
    # Use pre-computed masks when possible
    use_precomputed = (bits == _DEF_BITS)
    
    for tok in tokens:
        # Weight by token length with better scaling
        tok_len = len(tok)
        if tok_len <= 4:
            w = 1
        elif tok_len <= 8:
            w = 2
        elif tok_len <= 12:
            w = 3
        else:
            w = 4
        
        # Compute hash once
        h = int(hashlib.md5(tok.encode('utf-8', errors='ignore')).hexdigest(), 16)
        
        # Update vector with optimized bit operations
        if use_precomputed:
            for i in range(bits):
                if h & _BIT_MASKS[i]:
                    v[i] += w
                else:
                    v[i] -= w
        else:
            for i in range(bits):
                if h & (1 << i):
                    v[i] += w
                else:
                    v[i] -= w
    
    # Compute final hash
    out = 0
    if use_precomputed:
        for i in range(bits):
            if v[i] >= 0:
                out |= _BIT_MASKS[i]
    else:
        for i in range(bits):
            if v[i] >= 0:
                out |= (1 << i)
    
    # Return hex string with fixed width
    width = bits // 4
    return f"{out:0{width}x}"


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two simhash values.
    
    Returns number of differing bits.
    """
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return -1
    
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor = val1 ^ val2
        
        # Count set bits (Brian Kernighan's algorithm)
        count = 0
        while xor:
            xor &= xor - 1
            count += 1
        
        return count
    except ValueError:
        return -1


def is_similar(hash1: str, hash2: str, threshold: int = 3) -> bool:
    """Check if two hashes are similar within threshold.
    
    Args:
        hash1: First hash
        hash2: Second hash  
        threshold: Maximum Hamming distance for similarity (default: 3)
    
    Returns:
        True if hashes are similar
    """
    dist = hamming_distance(hash1, hash2)
    return 0 <= dist <= threshold
