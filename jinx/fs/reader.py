from __future__ import annotations

import os


def read_text(path: str) -> str:
    """Read entire text file if it exists else return empty string."""
    try:
        return open(path, encoding="utf-8").read().strip() if os.path.exists(path) else ""
    except Exception:
        # Maintain silent failure semantics similar to existing wire()
        return ""
