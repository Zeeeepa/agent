"""Vector-Based Semantic Search Stage - Pure embeddings, NO keywords.

Replaces: embeddings/project_stage_keyword.py

Uses:
- Semantic similarity only
- No token extraction
- No case-sensitive matching
- Pure vector operations
"""

from __future__ import annotations

import time
import numpy as np
from typing import List, Tuple, Dict, Any

from .project_retrieval_config import PROJ_MAX_FILES, PROJ_MAX_CHUNKS_PER_FILE
from .project_scan_store import iter_project_chunks


async def semantic_search(
    query: str,
    *,
    k: int = 20,
    max_time_ms: int = 250
) -> List[Tuple[float, str, Dict[str, Any]]]:
    """
    Pure semantic search using embeddings only.
    
    NO keyword extraction, NO token matching.
    Only vector similarity.
    
    Args:
        query: Query text
        k: Number of results
        max_time_ms: Time budget
    
    Returns:
        List of (score, file_rel, obj) tuples
    """
    
    if not query or not query.strip():
        return []
    
    try:
        from jinx.micro.embeddings.pipeline import embed_text
        # Get query embedding
        query_emb_obj = await embed_text(query, source='semantic_search', kind='query')
        if not isinstance(query_emb_obj, dict):
            return []
        emb = query_emb_obj.get('embedding')
        if not emb:
            return []
        query_vec = np.array(emb, dtype=np.float32)
        
    except Exception:
        return []
    
    t0 = time.perf_counter()
    
    results: List[Tuple[float, str, Dict[str, Any]]] = []
    
    # Iterate through project chunks
    for file_rel, obj in iter_project_chunks(
        max_files=PROJ_MAX_FILES,
        max_chunks_per_file=PROJ_MAX_CHUNKS_PER_FILE
    ):
        # Check timeout
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms >= max_time_ms:
            break
        
        # Get chunk embedding
        chunk_embedding = obj.get('embedding')
        
        if not chunk_embedding or not isinstance(chunk_embedding, (list, np.ndarray)):
            continue
        
        try:
            chunk_vec = np.array(chunk_embedding, dtype=np.float32)
            
            # Compute cosine similarity
            sim = _cosine_similarity(query_vec, chunk_vec)
            
            if sim > 0.1:  # Min threshold
                results.append((float(sim), file_rel, obj))
        
        except Exception:
            continue
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[0], reverse=True)
    
    return results[:k]


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity between vectors."""
    
    # Handle dimension mismatch
    if len(x) != len(y):
        min_dim = min(len(x), len(y))
        x = x[:min_dim]
        y = y[:min_dim]
    
    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
    
    return float(dot / (norm_x * norm_y))


__all__ = [
    "semantic_search",
]
