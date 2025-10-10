from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Thin facade only; heavy logic and caches live in micro-modules.


# Facade wiring: delegate heavy logic to micro-modules
from .retrieval_core import (
    retrieve_project_top_k as _core_retrieve_project_top_k,
    retrieve_project_multi_top_k as _core_retrieve_project_multi_top_k,
)
from .context_builder import (
    build_project_context_for as _core_build_project_context_for,
    build_project_context_multi_for as _core_build_project_context_multi_for,
)

__all__ = [
    "retrieve_project_top_k",
    "build_project_context_for",
    "retrieve_project_multi_top_k",
    "build_project_context_multi_for",
]


# Facade API: thin delegates

async def retrieve_project_top_k(query: str, k: int | None = None, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    return await _core_retrieve_project_top_k(query, k=k, max_time_ms=max_time_ms)


async def build_project_context_for(query: str, *, k: int | None = None, max_chars: int | None = None, max_time_ms: int | None = 300) -> str:
    # Facade: delegate to context_builder
    return await _core_build_project_context_for(query, k=k, max_chars=max_chars, max_time_ms=max_time_ms)
    

async def retrieve_project_multi_top_k(queries: List[str], *, per_query_k: int, max_time_ms: int | None = 300) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Facade: delegate to retrieval_core"""
    return await _core_retrieve_project_multi_top_k(queries, per_query_k=per_query_k, max_time_ms=max_time_ms)
    

async def build_project_context_multi_for(queries: List[str], *, k: int | None = None, max_chars: int | None = None, max_time_ms: int | None = 300) -> str:
    """Facade: delegate to context_builder"""
    return await _core_build_project_context_multi_for(queries, k=k, max_chars=max_chars, max_time_ms=max_time_ms)
    