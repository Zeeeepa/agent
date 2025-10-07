from __future__ import annotations

from typing import Optional
import ast
from .config import is_enabled


def check_nontrivial(code: str) -> Optional[str]:
    """Reject trivial blocks like bare string literals or metadata-only assignments.

    Rationale: The model may sometimes emit a plain English sentence as a bare
    string expression (valid Python but semantically useless), or assign only
    string constants to variables without any real logic. Such outputs should be
    treated as violations so that the recovery flow can trigger.
    """
    if not is_enabled("nontrivial", True):
        return None
    # Quick parse check; if syntax error, let this be reported as a violation
    try:
        tree = ast.parse(code or "")
    except SyntaxError as e:
        # Compact message to avoid leaking full source
        return f"syntax error: {e.msg} (line {getattr(e, 'lineno', '?')})"

    body = getattr(tree, "body", []) or []
    if not body:
        return "empty code block"

    def _is_bare_str_expr(node: ast.stmt) -> bool:
        return isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str)

    def _is_str_assign(node: ast.stmt) -> bool:
        if not isinstance(node, ast.Assign):
            return False
        val = getattr(node, "value", None)
        return isinstance(val, ast.Constant) and isinstance(val.value, str)

    # If the module consists only of bare string expressions and/or assignments of
    # string constants, consider it trivial.
    trivial = True
    for stmt in body:
        if _is_bare_str_expr(stmt):
            continue
        if _is_str_assign(stmt):
            continue
        trivial = False
        break

    if trivial:
        return "trivial code: avoid plain strings or metadata-only assignments; provide real Python logic"

    return None
