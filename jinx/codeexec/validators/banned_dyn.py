from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled
from .policy import BANNED_DYN_NAMES as _BANNED_NAMES, BANNED_DYN_ATTRS as _BANNED_ATTRS


def check_banned_dynamic(code: str) -> Optional[str]:
    """Reject dangerous dynamic evaluation/import patterns.

    Disallows eval/exec/compile/__import__, and importlib.import_module.
    """
    if not is_enabled("banned_dynamic", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            fn = getattr(n, "func", None)
            # direct name calls like eval(...)
            if isinstance(fn, ast.Name) and fn.id in _BANNED_NAMES:
                return f"dynamic call '{fn.id}(...)' is disallowed"
            # attribute calls like importlib.import_module(...)
            if isinstance(fn, ast.Attribute):
                mod = getattr(fn, "value", None)
                attr = fn.attr
                if isinstance(mod, ast.Name) and (mod.id, attr) in _BANNED_ATTRS:
                    return f"dynamic import '{mod.id}.{attr}(...)' is disallowed"
    return None
