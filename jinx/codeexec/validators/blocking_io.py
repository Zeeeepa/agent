from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled


def check_blocking_io(code: str) -> Optional[str]:
    """Disallow interactive/blocking input in RT context.

    Flags builtins.input(...) and sys.stdin.readline().
    """
    if not is_enabled("blocking_io", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            fn = getattr(n, "func", None)
            if isinstance(fn, ast.Name) and fn.id == "input":
                return "blocking input() is disallowed under RT constraints"
            if isinstance(fn, ast.Attribute) and fn.attr == "readline":
                v = getattr(fn, "value", None)
                if isinstance(v, ast.Attribute) and v.attr == "stdin":
                    u = getattr(v, "value", None)
                    if isinstance(u, ast.Name) and u.id == "sys":
                        return "blocking sys.stdin.readline() is disallowed under RT constraints"
    return None
