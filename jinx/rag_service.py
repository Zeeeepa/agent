from __future__ import annotations

import os
from typing import Dict, Any, List


ENV_OPENAI_VECTOR_STORE_ID: str = "OPENAI_VECTOR_STORE_ID"


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
    # Read single env var that may contain one or multiple comma-separated IDs.
    raw_ids = os.getenv(ENV_OPENAI_VECTOR_STORE_ID, "")
    vector_store_ids = _parse_vector_store_ids(raw_ids)

    if not vector_store_ids:
        return {}

    return {
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
            }
        ],
        # Force the model to call File Search instead of leaving it on auto.
        # "tool_choice": {"type": "file_search"},
    }
