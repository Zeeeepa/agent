from __future__ import annotations

import os
import json
from typing import Any, Dict

from jinx.micro.memory.storage import memory_dir


def facts_path() -> str:
    return os.path.join(memory_dir(), "facts.json")


def load_facts() -> Dict[str, Any]:
    try:
        with open(facts_path(), "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj  # type: ignore[return-value]
    except Exception:
        pass
    return {"paths": {}, "symbols": {}, "prefs": {}, "decisions": {}, "last_update_ts": 0}


def save_facts(data: Dict[str, Any]) -> None:
    try:
        os.makedirs(memory_dir(), exist_ok=True)
        with open(facts_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass
