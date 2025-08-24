from __future__ import annotations

import ast


def ast_normalize(code: str) -> str:
    """Parse and unparse Python code to normalize AST structure.

    Best-effort: raises nothing, returns original code on failure.
    """
    try:
        return ast.unparse(ast.parse(code))
    except Exception:
        return code
