from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled
from .policy import HEAVY_IMPORTS_TOP as _BANNED_TOP_LEVEL


def check_import_policy(code: str) -> Optional[str]:
    if not is_enabled("import_policy", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                top = (a.name or "").split(".")[0]
                if top in _BANNED_TOP_LEVEL:
                    return f"import of heavy module '{top}' is disallowed under RT constraints"
        if isinstance(n, ast.ImportFrom):
            top = (n.module or "").split(".")[0]
            if top in _BANNED_TOP_LEVEL:
                return f"from {top} import ... is disallowed under RT constraints"
    return None
