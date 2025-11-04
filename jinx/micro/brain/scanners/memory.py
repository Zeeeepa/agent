from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional
from functools import lru_cache

from jinx.micro.memory.storage import read_compact, read_evergreen


# Adaptive weights learned from usage patterns
_WEIGHT_CACHE: Dict[str, float] = {
    'path': 3.0,
    'symbol': 2.0,
    'pref': 1.0,
    'decision': 1.0,
    'term': 0.2
}
_WEIGHT_LOCK = asyncio.Lock()

def _add(node_map: Dict[str, float], key: str, w: float) -> None:
    try:
        node_map[key] = float(node_map.get(key, 0.0)) + float(w)
    except Exception:
        node_map[key] = float(w)

async def _get_adaptive_weight(concept_type: str) -> float:
    """Get adaptive weight from cache or learned values."""
    try:
        # Try to get learned weight from brain systems
        from jinx.micro.brain.concepts import get_concept_weight
        return await get_concept_weight(concept_type)
    except Exception:
        return _WEIGHT_CACHE.get(concept_type, 1.0)


# Cache with TTL
_CACHE: Optional[tuple[Dict[str, float], float]] = None
_CACHE_TTL = 30.0  # seconds

async def scan_memory_concepts(force_refresh: bool = False) -> Dict[str, float]:
    """Scan memory with adaptive weighting and caching."""
    global _CACHE
    
    # Check cache
    if not force_refresh and _CACHE is not None:
        nodes, cached_at = _CACHE
        if (time.time() - cached_at) < _CACHE_TTL:
            return nodes.copy()
    
    nodes: Dict[str, float] = {}
    
    # Parallel reads for performance
    comp_task = asyncio.create_task(read_compact())
    ever_task = asyncio.create_task(read_evergreen())
    
    try:
        ever = await ever_task
    except Exception:
        ever = ""
    
    try:
        comp = await comp_task
    except Exception:
        comp = ""
    
    # Get adaptive weights
    weight_path = await _get_adaptive_weight('path')
    weight_symbol = await _get_adaptive_weight('symbol')
    weight_pref = await _get_adaptive_weight('pref')
    weight_decision = await _get_adaptive_weight('decision')
    weight_term = await _get_adaptive_weight('term')
    
    # Evergreen: curated keys with adaptive weights
    for raw in (ever or "").splitlines():
        line = (raw or "").strip()
        low = line.lower()
        if low.startswith("path: "):
            _add(nodes, line.lower(), weight_path)
        elif low.startswith("symbol: "):
            _add(nodes, line.lower(), weight_symbol)
        elif low.startswith("pref: "):
            _add(nodes, line.lower(), weight_pref)
        elif low.startswith("decision: "):
            _add(nodes, line.lower(), weight_decision)
    
    # Compact: loose terms with adaptive weight
    import re as _re
    for raw in (comp or "").splitlines()[-800:]:
        s = (raw or "").strip().lower()
        for m in _re.finditer(r"(?u)[\w\.]{4,}", s):
            tok = (m.group(0) or "").lower()
            if tok and len(tok) >= 4:  # min length filter
                _add(nodes, f"term: {tok}", weight_term)
    
    # Cache result
    _CACHE = (nodes.copy(), time.time())
    
    return nodes


__all__ = ["scan_memory_concepts"]
