from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled


_UNSAFE_LOAD_FUNCS = {
    ("pickle", "load"),
    ("pickle", "loads"),
    ("dill", "load"),
    ("dill", "loads"),
    ("marshal", "load"),
    ("marshal", "loads"),
}


def check_deserialization_safety(code: str) -> Optional[str]:
    """Disallow unsafe deserialization primitives.

    - Ban pickle/dill/marshal load/loads
    - Forbid yaml.load; require yaml.safe_load instead
    """
    if not is_enabled("deserialization_safety", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            fn = getattr(n, "func", None)
            # module.func calls
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                mod = fn.value.id
                attr = fn.attr
                if (mod, attr) in _UNSAFE_LOAD_FUNCS:
                    return f"unsafe deserialization '{mod}.{attr}(...)' is disallowed"
                if mod == "yaml" and attr == "load":
                    return "yaml.load is disallowed; use yaml.safe_load"
    return None
