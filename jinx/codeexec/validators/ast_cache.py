from __future__ import annotations

from typing import Optional, Tuple
import ast

# Tiny per-process cache to avoid reparsing the same code multiple times
# during a single validation pass. Not thread-safe in general, but good
# enough for our single-pass validator usage.
_last: Tuple[str, Optional[ast.AST]] | None = None


def get_ast(code: str) -> Optional[ast.AST]:
    global _last
    s = code or ""
    if _last and _last[0] == s:
        return _last[1]
    try:
        tree = ast.parse(s)
    except SyntaxError:
        tree = None
    _last = (s, tree)
    return tree
