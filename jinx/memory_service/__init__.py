from __future__ import annotations

"""Memory Service facade.

Thin re-export of the micro memory service contract.
"""

from jinx.micro.memory_service.service import (
    MemoryService as MemoryService,
    start_memory_service_task as start_memory_service_task,
)

__all__ = [
    "MemoryService",
    "start_memory_service_task",
]
