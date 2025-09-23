from __future__ import annotations

from .pipeline import embed_text, iter_recent_items
from .retrieval import retrieve_top_k, build_context_for
from .service import EmbeddingsService, start_embeddings_task

__all__ = [
    "embed_text",
    "iter_recent_items",
    "retrieve_top_k",
    "build_context_for",
    "EmbeddingsService",
    "start_embeddings_task",
]
