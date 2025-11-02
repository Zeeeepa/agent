from __future__ import annotations

import os

# Internal brain workspace (separate from emb/ and log/)
BRAIN_ROOT = os.path.join(".jinx", "brain")
CONCEPTS_PATH = os.path.join(BRAIN_ROOT, "concepts.json")


def ensure_brain_dirs() -> None:
    try:
        os.makedirs(BRAIN_ROOT, exist_ok=True)
    except Exception:
        pass
