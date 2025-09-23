from __future__ import annotations

from typing import Any, Dict

# Facade that delegates to micro-module implementation.
# Keep the old import path stable for callers.
from jinx.micro.rag.file_search import (
    ENV_OPENAI_VECTOR_STORE_ID,
    ENV_OPENAI_FORCE_FILE_SEARCH,
    build_file_search_tools as _build_file_search_tools,
)


def build_file_search_tools() -> Dict[str, Any]:
    """Compatibility facade for File Search tool binding.

    Delegates to ``jinx.micro.rag.file_search.build_file_search_tools`` to keep
    the legacy import location stable while the logic lives in the micro-module.
    """
    return _build_file_search_tools()


__all__ = [
    "ENV_OPENAI_VECTOR_STORE_ID",
    "ENV_OPENAI_FORCE_FILE_SEARCH",
    "build_file_search_tools",
]
