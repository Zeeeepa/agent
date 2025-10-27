from __future__ import annotations

from .service import (
    EmbedderService as EmbedderService,
    start_embedder_service_task as start_embedder_service_task,
)

__all__ = [
    "EmbedderService",
    "start_embedder_service_task",
]
