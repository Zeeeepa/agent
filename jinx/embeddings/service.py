from __future__ import annotations

"""Embeddings service facade.

Delegates to the micro-module implementation under
``jinx.micro.embeddings.service`` while keeping the public API stable.
"""

from jinx.micro.embeddings.service import (
    EmbeddingsService as EmbeddingsService,
    start_embeddings_task as start_embeddings_task,
    stop_embeddings_task as stop_embeddings_task,
)
from jinx.micro.embeddings.project_service import (
    ProjectEmbeddingsService as ProjectEmbeddingsService,
    start_project_embeddings_task as start_project_embeddings_task,
    stop_project_embeddings_task as stop_project_embeddings_task,
)


__all__ = [
    "EmbeddingsService",
    "start_embeddings_task",
    "stop_embeddings_task",
    "ProjectEmbeddingsService",
    "start_project_embeddings_task",
    "stop_project_embeddings_task",
]
