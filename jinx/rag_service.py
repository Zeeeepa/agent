from __future__ import annotations

from typing import Any, Dict

# Facade that delegates to micro-module implementation.
# Keep the old import path stable for callers.
from jinx.micro.rag.file_search import (
    ENV_OPENAI_VECTOR_STORE_ID,
    ENV_OPENAI_FORCE_FILE_SEARCH,
    build_file_search_tools as _build_file_search_tools,
)
from jinx.settings import Settings


def build_file_search_tools() -> Dict[str, Any]:
    """Compatibility facade for File Search tool binding.

    Delegates to ``jinx.micro.rag.file_search.build_file_search_tools`` to keep
    the legacy import location stable while the logic lives in the micro-module.
    """
    return _build_file_search_tools()


def build_file_search_tools_from_settings(settings: Settings) -> Dict[str, Any]:
    """Construct File Search tool kwargs directly from Settings.

    Mirrors the micro-module layout while avoiding env access at call sites.
    """
    ids = settings.openai.vector_store_ids
    if not ids:
        return {}
    extra: Dict[str, Any] = {
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": ids,
            }
        ]
    }
    if settings.openai.force_file_search:
        extra["tool_choice"] = {"type": "file_search"}
    return extra


__all__ = [
    "ENV_OPENAI_VECTOR_STORE_ID",
    "ENV_OPENAI_FORCE_FILE_SEARCH",
    "build_file_search_tools",
    "build_file_search_tools_from_settings",
]
