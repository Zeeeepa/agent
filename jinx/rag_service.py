from __future__ import annotations

import os
from typing import Dict, Any


def build_file_search_tools() -> Dict[str, Any]:
    """Return extra kwargs for OpenAI Responses API to enable File Search.

    Behavior:
    - If OPENAI_VECTOR_STORE_ID is set and non-empty -> return tools payload
      binding that vector store.
    - Otherwise -> return empty dict (feature off by default).
    """
    vs_id = (os.getenv("OPENAI_VECTOR_STORE_ID") or "").strip()
    if not vs_id:
        return {}

    return {
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [vs_id],
            }
        ]
    }
