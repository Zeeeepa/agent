"""Advanced term extraction with caching and optimization."""

from __future__ import annotations

import re
import importlib
from typing import List, Dict, Set, Tuple
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class TermScore:
    """Term with its computed score."""
    term: str
    frequency: int
    line_coverage: float
    score: float


@lru_cache(maxsize=128)
def _compile_word_regex() -> re.Pattern:
    """Compile and cache word regex."""
    return re.compile(r"(?u)[\w]+")


def extract_terms(text: str, top_k: int = 25) -> List[str]:
    """Language-agnostic term extractor without stopwords.

    Approach:
    - Tokenize with Unicode-aware word regex (\\w).
    - Keep tokens with at least one letter and length >= 3; drop digits-only.
    - Score by frequency discounted by line coverage (proxy for IDF within the document):
        score = tf * (1 - line_occurrence_ratio)
      where line_occurrence_ratio is fraction of non-empty lines containing the token.
    - Small bonus for identifier-like tokens (underscore or digits in token).
    - Deterministic tie-breaker by token string.
    """
    # Optional plugin hook: if a module 'jinx_terms_plugin' provides extract_terms, use it.
    try:
        _mod = importlib.import_module("jinx_terms_plugin")
        _fn = getattr(_mod, "extract_terms", None)
        if callable(_fn):
            out = _fn(text, top_k)  # type: ignore[call-arg]
            if isinstance(out, list):
                # Trust plugin result if structurally valid
                return [str(x) for x in out][: top_k]
    except Exception:
        pass

    text = text or ""
    if not text.strip():
        return []

    # Split into lines for line-occurrence stats
    lines = text.splitlines()
    non_empty_lines = [ln for ln in lines if ln.strip()]
    total_lines = max(1, len(non_empty_lines))

    # Collect term frequencies and line-level occurrences
    tf: Dict[str, int] = {}
    line_occ: Dict[str, int] = {}

    # Use cached regex
    word_re = _compile_word_regex()

    # Optimized loop with early filtering
    for ln in non_empty_lines:
        seen_in_line: Set[str] = set()
        
        for m in word_re.finditer(ln):
            w_raw = m.group(0)
            
            # Early length check
            if len(w_raw) < 3:
                continue
            
            # Require at least one alphabetic character
            if not any(ch.isalpha() for ch in w_raw):
                continue
            
            w = w_raw.lower()
            
            # Update frequency
            tf[w] = tf.get(w, 0) + 1
            
            # Track line occurrence (for IDF proxy)
            if w not in seen_in_line:
                line_occ[w] = line_occ.get(w, 0) + 1
                seen_in_line.add(w)

    if not tf:
        return []

    # Compute scores efficiently
    scored_terms: List[Tuple[str, float]] = []
    
    for w, freq in tf.items():
        # Line coverage (IDF proxy)
        line_freq = line_occ.get(w, 0) / float(total_lines)
        
        # Base score: TF * (1 - line_coverage)
        base = float(freq) * max(0.0, 1.0 - line_freq)
        
        # Identifier-like bonus (underscores or digits = more specific)
        if "_" in w or any(ch.isdigit() for ch in w):
            base *= 1.1
        
        # CamelCase bonus (likely identifier)
        if any(ch.isupper() for ch in w) and any(ch.islower() for ch in w):
            base *= 1.05
        
        # Only keep terms with meaningful scores
        if base > 0.0:
            scored_terms.append((w, base))
    
    # Sort by score (descending) then alphabetically
    scored_terms.sort(key=lambda x: (-x[1], x[0]))
    
    # Return top K terms
    return [w for (w, _) in scored_terms[:top_k]]


def extract_terms_detailed(text: str, top_k: int = 25) -> List[TermScore]:
    """Extract terms with detailed scoring information.
    
    Returns list of TermScore objects with full metadata.
    """
    # Reuse main logic but return detailed scores
    text = text or ""
    if not text.strip():
        return []
    
    lines = text.splitlines()
    non_empty_lines = [ln for ln in lines if ln.strip()]
    total_lines = max(1, len(non_empty_lines))
    
    tf: Dict[str, int] = {}
    line_occ: Dict[str, int] = {}
    word_re = _compile_word_regex()
    
    for ln in non_empty_lines:
        seen_in_line: Set[str] = set()
        for m in word_re.finditer(ln):
            w_raw = m.group(0)
            if len(w_raw) < 3 or not any(ch.isalpha() for ch in w_raw):
                continue
            w = w_raw.lower()
            tf[w] = tf.get(w, 0) + 1
            if w not in seen_in_line:
                line_occ[w] = line_occ.get(w, 0) + 1
                seen_in_line.add(w)
    
    if not tf:
        return []
    
    results: List[TermScore] = []
    for w, freq in tf.items():
        line_coverage = line_occ.get(w, 0) / float(total_lines)
        base = float(freq) * max(0.0, 1.0 - line_coverage)
        
        if "_" in w or any(ch.isdigit() for ch in w):
            base *= 1.1
        
        if base > 0.0:
            results.append(TermScore(
                term=w,
                frequency=freq,
                line_coverage=line_coverage,
                score=base
            ))
    
    results.sort(key=lambda x: (-x.score, x.term))
    return results[:top_k]
