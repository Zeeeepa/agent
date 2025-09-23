from __future__ import annotations

"""Embeddings retrieval facade.

Delegates to the micro-module implementation under
``jinx.micro.embeddings.retrieval`` while keeping the public API stable.
"""

from jinx.micro.embeddings.retrieval import (
    retrieve_top_k as retrieve_top_k,
    build_context_for as build_context_for,
)


__all__ = [
    "retrieve_top_k",
    "build_context_for",
]
