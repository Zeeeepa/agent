from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled, HTTP_MAX_TIMEOUT as _MAX_TIMEOUT


def _const_num(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _kw_value(call: ast.Call, name: str) -> Optional[ast.AST]:
    for kw in call.keywords or []:
        if (kw.arg or "") == name:
            return kw.value
    return None


def _is_requests_call(func: ast.AST) -> bool:
    # requests.get/post/put/delete/head/options
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "requests"
        and func.attr in {"get", "post", "put", "delete", "head", "options", "patch"}
    )


def _is_urlopen_call(func: ast.AST) -> bool:
    # urllib.request.urlopen
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "urllib"
        and func.value.attr == "request"
        and func.attr == "urlopen"
    )


def check_http_safety(code: str) -> Optional[str]:
    if not is_enabled("http_safety", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            fn = n.func
            if _is_requests_call(fn) or _is_urlopen_call(fn):
                # Require timeout kw and clamp
                tv = _kw_value(n, "timeout")
                if tv is None:
                    return "network call requires explicit timeout <= 10s"
                v = _const_num(tv)
                if v is not None and v > _MAX_TIMEOUT:
                    return f"timeout too large: {v}s (> {_MAX_TIMEOUT}s)"
                # For requests: forbid verify=False (allow verify=True or default)
                if _is_requests_call(fn):
                    vv = _kw_value(n, "verify")
                    if isinstance(vv, ast.Constant) and vv.value is False:
                        return "requests.* with verify=False is disallowed"
    return None
