from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled


def check_spawn_policy(code: str) -> Optional[str]:
    """Disallow ad-hoc thread/process/eventloop spawns; use runtime primitives instead.

    Flags:
    - threading.Thread(...)
    - multiprocessing.Process(...)
    - asyncio.run(...)
    - *.run_forever() on event loops
    """
    if not is_enabled("spawn_policy", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            fn = n.func
            # threading.Thread(...)
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "threading" and fn.attr == "Thread":
                return "thread spawns are disallowed; use micro runtime primitives (spawn/submit_task)"
            # multiprocessing.Process(...)
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "multiprocessing" and fn.attr == "Process":
                return "process spawns are disallowed; use micro runtime primitives"
            # asyncio.run(...)
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "asyncio" and fn.attr == "run":
                return "asyncio.run() is disallowed; integrate with the existing loop via runtime APIs"
            # loop.run_forever()
            if isinstance(fn, ast.Attribute) and fn.attr == "run_forever":
                return "run_forever() is disallowed under RT constraints"
    return None
