from __future__ import annotations

"""Embedder Service facade.

Re-exports the micro-module contract to keep facades thin.
"""

from jinx.micro.embedder_service.service import (
    EmbedderService as EmbedderService,
    start_embedder_service_task as start_embedder_service_task,
)

__all__ = [
    "EmbedderService",
    "start_embedder_service_task",
]
