from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled


def check_try_except_ast(code: str) -> Optional[str]:
    """Detect Python try/except/finally via AST (precise).
    Returns a violation if any ast.Try node is present.
    """
    if not is_enabled("try_except", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Try):
            return "Usage of try/except/finally is not allowed by prompt"
    return None
