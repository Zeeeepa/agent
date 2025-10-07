from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled, IO_MAX_LOOP_BODY_LINES as _MAX_LOOP_BODY_LINES, IO_MAX_LITERAL_ELEMS as _MAX_LIST_LITERAL_ELEMS


def check_io_clamps(code: str) -> Optional[str]:
    """Clamp obviously runaway patterns to preserve RT constraints.

    Flags:
    - loops whose body spans too many lines
    - gigantic list/dict/set literals (likely model dumping data)
    """
    if not is_enabled("io_clamps", True):
        return None
    t = get_ast(code)
    if not t:
        return None

    def _span(n: ast.AST) -> int:
        a = int(getattr(n, "lineno", 0) or 0)
        b = int(getattr(n, "end_lineno", a) or a)
        return max(0, b - a)

    for n in ast.walk(t):
        if isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
            body_span = sum(_span(ch) for ch in (n.body or []))
            if body_span > _MAX_LOOP_BODY_LINES:
                return f"loop body too large: ~{body_span} lines"
        if isinstance(n, (ast.List, ast.Set, ast.Tuple)):
            if len(getattr(n, "elts", []) or []) > _MAX_LIST_LITERAL_ELEMS:
                return f"literal too large: ~{len(n.elts)} elements"
        if isinstance(n, ast.Dict):
            if len(getattr(n, "keys", []) or []) > _MAX_LIST_LITERAL_ELEMS:
                return f"dict literal too large: ~{len(n.keys)} keys"
    return None
