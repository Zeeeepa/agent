from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable, List, Tuple


CHARS_PER_CHUNK = int(os.getenv("EMBED_PROJECT_CHARS_PER_CHUNK", "1200"))
MIN_CHUNK_CHARS = int(os.getenv("EMBED_PROJECT_MIN_CHUNK_CHARS", "150"))
MAX_CHUNKS_PER_FILE = int(os.getenv("EMBED_PROJECT_MAX_CHUNKS_PER_FILE", "200"))
TOP_TERMS = int(os.getenv("EMBED_PROJECT_TOP_TERMS", "25"))


def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


def tokenize_terms(text: str, top_k: int = TOP_TERMS) -> List[str]:
    freq: dict[str, int] = {}
    # Use Unicode-aware word extraction: split on non-letter boundaries, ignore digits-only tokens
    for m in re.finditer(r"(?u)[\w]+", text):
        w = m.group(0)
        if not any(ch.isalpha() for ch in w):
            continue
        w = w.lower()
        if len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    # Sort by count desc then alphabetically to keep stable
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in items[: top_k]]


def build_chunks(text: str, *, max_chars: int = CHARS_PER_CHUNK, max_chunks: int = MAX_CHUNKS_PER_FILE) -> List[str]:
    """Split text into rough max_chars chunks preserving lines.

    Avoids heavy tokenization dependencies; chars ~= tokens*4.
    """
    if not text:
        return []
    lines = text.splitlines()
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for ln in lines:
        ln2 = ln.rstrip("\n")
        l = len(ln2) + 1  # keep newline budget
        if cur_len + l > max_chars and cur_len >= MIN_CHUNK_CHARS:
            chunks.append("\n".join(cur).strip())
            if len(chunks) >= max_chunks:
                break
            cur = []
            cur_len = 0
        cur.append(ln2)
        cur_len += l
    if cur and len(chunks) < max_chunks:
        s = "\n".join(cur).strip()
        if s:
            chunks.append(s)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS or len(chunks) == 1]


def file_should_include(path: str, *, include_exts: Iterable[str], exclude_dirs: Iterable[str]) -> bool:
    p = os.path.normpath(path)
    parts = p.split(os.sep)
    for d in exclude_dirs:
        if d and d in parts:
            return False
    ext = os.path.splitext(p)[1].lower().lstrip(".")
    return (not include_exts) or (ext in include_exts)
