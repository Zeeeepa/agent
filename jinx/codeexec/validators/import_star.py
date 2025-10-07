from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled


def check_import_star(code: str) -> Optional[str]:
    """Disallow 'from x import *' which harms determinism and clarity."""
    if not is_enabled("import_star", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom):
            for alias in (n.names or []):
                if alias.name == "*":
                    mod = n.module or "<module>"
                    return f"from {mod} import * is disallowed"
    return None
