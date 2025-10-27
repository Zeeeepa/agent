from __future__ import annotations

import hashlib
from typing import Iterable

# Simple simhash-like fingerprint for text chunks to support dedup/consolidation.
# Not cryptographic. Produces a fixed-length hex string (64-bit by default).

_DEF_BITS = 64


def _tokens(text: str) -> Iterable[str]:
    t = (text or "").lower()
    # crude tokenization: split on non-alnum, keep identifiers and words
    cur = []
    for ch in t:
        if ch.isalnum() or ch == '_':
            cur.append(ch)
        else:
            if cur:
                yield "".join(cur)
                cur = []
    if cur:
        yield "".join(cur)


def simhash(text: str, *, bits: int = _DEF_BITS) -> str:
    if not text:
        return "0" * (bits // 4)
    v = [0] * bits
    for tok in _tokens(text):
        # weight by token length; could add idf in the future
        w = max(1, min(5, len(tok) // 4))
        h = int(hashlib.md5(tok.encode('utf-8', errors='ignore')).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += w
            else:
                v[i] -= w
    out = 0
    for i in range(bits):
        if v[i] >= 0:
            out |= (1 << i)
    # hex string, fixed width
    width = bits // 4
    return f"{out:0{width}x}"
