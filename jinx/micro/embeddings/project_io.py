from __future__ import annotations

import json
import os
from typing import Any


def write_json_atomic(path: str, obj: Any) -> None:
    """Atomically write JSON to path using a temporary file and os.replace.

    Best-effort: cleans up temporary file on error.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    # Prefer orjson if available for speed; fallback to json
    try:
        import orjson  # type: ignore
        text = orjson.dumps(obj).decode("utf-8")
    except Exception:
        text = json.dumps(obj, ensure_ascii=False)
    try:
        with open(tmp, "w", encoding="utf-8") as w:
            w.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
