from __future__ import annotations

import os
from typing import List
from .project_chunk_types import Chunk

# Token-based chunking parameters
TOKENS_PER_CHUNK = int(os.getenv("EMBED_PROJECT_TOKENS_PER_CHUNK", "300"))
MIN_CHUNK_TOKENS = int(os.getenv("EMBED_PROJECT_MIN_CHUNK_TOKENS", "40"))
MAX_CHUNKS_PER_FILE = int(os.getenv("EMBED_PROJECT_MAX_CHUNKS_PER_FILE", "200"))
# Small token overlap for better boundary recall
TOK_OVERLAP = int(os.getenv("EMBED_PROJECT_TOKEN_OVERLAP", "32"))


def _tiktoken_encode(text: str):
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None, None
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            return None, None
    return enc, enc.encode(text)


def chunk_text_token(text: str) -> List[Chunk]:
    """Split text by tokens using tiktoken if available; otherwise empty list.

    Returns list of dicts {"text": str, "line_start": int, "line_end": int}.
    For token chunker, line indices are set to 0 (unknown).
    """
    if not text:
        return []
    enc, toks = _tiktoken_encode(text)
    if enc is None or toks is None:
        return []
    chunks: List[Chunk] = []
    n = len(toks)
    i = 0
    while i < n and len(chunks) < MAX_CHUNKS_PER_FILE:
        j = min(n, i + TOKENS_PER_CHUNK)
        sub = toks[i:j]
        if len(sub) < MIN_CHUNK_TOKENS and chunks:
            break
        try:
            sub_text = enc.decode(sub).strip()
        except Exception:
            sub_text = ""
        if sub_text:
            chunks.append(Chunk(text=sub_text, line_start=0, line_end=0))
        # Advance with overlap
        if TOK_OVERLAP > 0:
            i = max(i + 1, j - TOK_OVERLAP)
        else:
            i = j
    return chunks
