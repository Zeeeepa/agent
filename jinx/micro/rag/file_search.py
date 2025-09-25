from __future__ import annotations

import os
from typing import Any, Dict, List

# Env keys are part of the micro-module contract
ENV_OPENAI_VECTOR_STORE_ID: str = "OPENAI_VECTOR_STORE_ID"
ENV_OPENAI_FORCE_FILE_SEARCH: str = "OPENAI_FORCE_FILE_SEARCH"


def _is_on(val: str | None) -> bool:
    return (val or "0").strip().lower() in {"1", "true", "yes", "on"}


def _parse_vector_store_ids(raw: str) -> List[str]:
    """Parse a comma-separated list of vector store IDs.

    - Trims whitespace around IDs
    - Drops empty entries
    - Deduplicates while preserving order
    """
    if not raw:
        return []

    ids: List[str] = [i.strip() for i in raw.split(",") if i.strip()]
    # Deduplicate while preserving order
    return list(dict.fromkeys(ids)) if ids else []


def build_file_search_tools() -> Dict[str, Any]:
    """Return extra kwargs for OpenAI Responses API to enable File Search.

    Behavior:
    - If OPENAI_VECTOR_STORE_ID is set (single or comma-separated) -> bind those.
    - Otherwise -> return empty dict (feature off by default).
    """
    raw_ids = os.getenv(ENV_OPENAI_VECTOR_STORE_ID, "")
    vector_store_ids = _parse_vector_store_ids(raw_ids)

    # Force File Search by default when vector_store_ids are present; can be disabled via env
    force_file_search = _is_on(os.getenv(ENV_OPENAI_FORCE_FILE_SEARCH, "1"))

    if not vector_store_ids:
        return {}

    extra: Dict[str, Any] = {
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
            }
        ]
    }

    # If enabled via env, force the model to call File Search instead of leaving it on auto.
    if force_file_search:
        extra["tool_choice"] = {"type": "file_search"}

    return extra


__all__ = [
    "ENV_OPENAI_VECTOR_STORE_ID",
    "ENV_OPENAI_FORCE_FILE_SEARCH",
    "_parse_vector_store_ids",
    "build_file_search_tools",
]
