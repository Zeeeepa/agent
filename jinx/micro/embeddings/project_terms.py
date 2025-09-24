from __future__ import annotations

import re
import importlib
from typing import List


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
    tf: dict[str, int] = {}
    line_occ: dict[str, int] = {}

    # Precompile regex once
    word_re = re.compile(r"(?u)[\w]+")

    for ln in non_empty_lines:
        seen_in_line: set[str] = set()
        for m in word_re.finditer(ln):
            w_raw = m.group(0)
            # Require at least one alphabetic character to avoid pure numeric/punct
            if not any(ch.isalpha() for ch in w_raw):
                continue
            w = w_raw.lower()
            if len(w) < 3:
                continue
            tf[w] = tf.get(w, 0) + 1
            if w not in seen_in_line:
                line_occ[w] = line_occ.get(w, 0) + 1
                seen_in_line.add(w)

    if not tf:
        return []

    # Compute scores with a simple per-line discount; add identifier-like bonus
    def score_of(w: str) -> float:
        lf = line_occ.get(w, 0) / float(total_lines)
        base = float(tf.get(w, 0)) * max(0.0, 1.0 - lf)
        # Identifier-like bonus: underscores or digits indicate specificity in code contexts
        if ("_" in w) or any(ch.isdigit() for ch in w):
            base *= 1.1
        return base

    # Rank tokens; filter near-zero scores
    items = [(w, score_of(w)) for w in tf.keys()]
    items = [(w, s) for (w, s) in items if s > 0.0]
    items.sort(key=lambda x: (-x[1], x[0]))
    return [w for (w, _) in items[: top_k]]
