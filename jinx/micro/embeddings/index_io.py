from __future__ import annotations

import asyncio
import json
import os

from .paths import INDEX_DIR


async def append_index(source: str, row: dict) -> None:
    """Append a JSONL row to the per-source index file.

    Best-effort semantics with a short retry loop.
    """
    safe_source = source.replace(os.sep, "__").replace("/", "__")
    path = os.path.join(INDEX_DIR, f"{safe_source}.jsonl")
    line = json.dumps(row, ensure_ascii=False)
    for _ in range(3):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return
        except Exception:
            await asyncio.sleep(0.05)
    # Drop silently on persistent failure
