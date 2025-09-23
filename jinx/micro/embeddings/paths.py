from __future__ import annotations

import os

EMBED_ROOT = os.path.join("log", "embeddings")
INDEX_DIR = os.path.join(EMBED_ROOT, "index")


def ensure_dirs() -> None:
    os.makedirs(EMBED_ROOT, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
