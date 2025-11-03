from __future__ import annotations

import os
from typing import List
from .project_chunk_types import Chunk

# Character-based chunking parameters
CHARS_PER_CHUNK = int(os.getenv("EMBED_PROJECT_CHARS_PER_CHUNK", "1200"))
MIN_CHUNK_CHARS = int(os.getenv("EMBED_PROJECT_MIN_CHUNK_CHARS", "150"))
MAX_CHUNKS_PER_FILE = int(os.getenv("EMBED_PROJECT_MAX_CHUNKS_PER_FILE", "200"))
# Small overlap in lines between adjacent chunks to preserve boundary context
OVERLAP_LINES = int(os.getenv("EMBED_PROJECT_OVERLAP_LINES", "8"))


def chunk_text_char(text: str) -> List[Chunk]:
    """Split text into chunks ~CHARS_PER_CHUNK preserving line boundaries.

    Returns a list of dicts: {"text": str, "line_start": int, "line_end": int}
    where line indices are 1-based inclusive.
    """
    if not text:
        return []
    lines = text.splitlines()
    chunks: List[Chunk] = []
    cur_lines: List[str] = []
    cur_len = 0
    start_line = 1
    for i, ln in enumerate(lines, start=1):
        ln2 = ln.rstrip("\n")
        l = len(ln2) + 1  # count newline budget
        if cur_len + l > CHARS_PER_CHUNK and cur_len >= MIN_CHUNK_CHARS:
            chunk_text = "\n".join(cur_lines).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "line_start": start_line, "line_end": i - 1})
            if len(chunks) >= MAX_CHUNKS_PER_FILE:
                break
            # Start next chunk with a small overlap from the end of previous chunk
            if OVERLAP_LINES > 0 and cur_lines:
                tail = cur_lines[-min(OVERLAP_LINES, len(cur_lines)) :]
            else:
                tail = []
            cur_lines = list(tail)
            cur_len = sum(len(x) + 1 for x in cur_lines)
            # Adjust start_line backward by overlap amount (but not before previous start)
            start_line = max(1, (i - len(cur_lines)))
        cur_lines.append(ln2)
        cur_len += l
    if cur_lines and len(chunks) < MAX_CHUNKS_PER_FILE:
        chunk_text = "\n".join(cur_lines).strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "line_start": start_line, "line_end": len(lines)})
    # Allow single short chunk if whole file is short
    if len(chunks) == 1 and len(chunks[0]["text"]) < MIN_CHUNK_CHARS:
        return chunks
    # Otherwise drop tiny trailing chunk
    return [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]
