from __future__ import annotations

"""Project embeddings retrieval facade.

Delegates to the micro-module implementation under
``jinx.micro.embeddings.project_retrieval`` while keeping the public API stable.
"""

from jinx.micro.embeddings.project_retrieval import (
    retrieve_project_top_k as retrieve_project_top_k,
    build_project_context_for as build_project_context_for,
)

__all__ = [
    "retrieve_project_top_k",
    "build_project_context_for",
]
