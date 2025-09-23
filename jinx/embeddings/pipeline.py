from __future__ import annotations

"""Embeddings pipeline facade.

Delegates to the micro-module implementation under
``jinx.micro.embeddings.pipeline`` while keeping the public API stable.
"""

from jinx.micro.embeddings.pipeline import (
    embed_text as embed_text,
    iter_recent_items as iter_recent_items,
)


__all__ = [
    "embed_text",
    "iter_recent_items",
]
