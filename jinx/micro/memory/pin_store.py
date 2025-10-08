from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from jinx.micro.memory.storage import memory_dir

_PIN_PATH = os.path.join(memory_dir(), "pinned.json")


def load_pins() -> List[str]:
    try:
        with open(_PIN_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                return [str(x) for x in obj]
    except Exception:
        pass
    return []


def save_pins(items: List[str]) -> None:
    try:
        os.makedirs(memory_dir(), exist_ok=True)
        with open(_PIN_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
    except Exception:
        pass


def is_pinned(line: str) -> bool:
    try:
        pins = load_pins()
        return (line or "").strip() in pins
    except Exception:
        return False
