from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled


def check_side_effect_policy(code: str) -> Optional[str]:
    """Flag direct UI-launching OS side effects, independent of prompts.

    Disallows:
    - webbrowser.open(...)
    - os.startfile(...)

    Rationale: these trigger system/UI actions that are not universally safe in
    sandboxed or CI contexts. Prefer returning the target (URL/path) or using
    higher-level runtime primitives to delegate the action.
    """
    if not is_enabled("side_effects", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            fn = getattr(n, "func", None)
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                # webbrowser.open(...)
                if fn.value.id == "webbrowser" and fn.attr == "open":
                    return "UI side-effect 'webbrowser.open(...)' is disallowed; return the URL or delegate via runtime primitives"
                # os.startfile(...)
                if fn.value.id == "os" and fn.attr == "startfile":
                    return "UI side-effect 'os.startfile(...)' is disallowed; return the path or delegate via runtime primitives"
    return None
