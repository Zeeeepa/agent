from __future__ import annotations

import os
from typing import Callable, Dict

# Registry for prompt providers: name -> loader function returning the prompt string
_REGISTRY: Dict[str, Callable[[], str]] = {}


def register_prompt(name: str, loader: Callable[[], str]) -> None:
    name = name.strip().lower()
    _REGISTRY[name] = loader


def get_prompt(name: str | None = None) -> str:
    """Return the prompt text by name.

    If name is None, resolve from environment variable JINX_PROMPT (or PROMPT_NAME),
    defaulting to "burning_logic".
    """
    if not name:
        name = os.getenv("JINX_PROMPT") or os.getenv("PROMPT_NAME") or "burning_logic"
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown prompt '{name}'. Available: {available}")
    return _REGISTRY[key]()


# Import built-ins to register them
from . import burning_logic  # noqa: F401
from . import chaos_bloom  # noqa: F401
from . import jinxed_blueprint  # noqa: F401
from . import memory_optimizer  # noqa: F401
from . import burning_logic_recovery  # noqa: F401
from . import planner_minjson  # noqa: F401
from . import planner_reflectjson  # noqa: F401
from . import planner_advisoryjson  # noqa: F401
from . import planner_reflectadvisoryjson  # noqa: F401
