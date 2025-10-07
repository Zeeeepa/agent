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


def _is_httpx_direct_call(func: ast.AST) -> bool:
    # httpx.get/post/put/delete/head/options/patch/request
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "httpx"
        and func.attr in {"get", "post", "put", "delete", "head", "options", "patch", "request"}
    )


def _is_httpx_client_ctor(func: ast.AST) -> bool:
    # httpx.Client(...) or httpx.AsyncClient(...)
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "httpx"
        and func.attr in {"Client", "AsyncClient"}
    )


def _is_aiohttp_request(func: ast.AST) -> bool:
    # aiohttp.request(...)
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "aiohttp"
        and func.attr == "request"
    )


def _is_aiohttp_session_ctor(func: ast.AST) -> bool:
    # aiohttp.ClientSession(...)
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "aiohttp"
        and func.attr == "ClientSession"
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
            # Direct HTTP calls require short timeouts and safe TLS settings
            if _is_requests_call(fn) or _is_urlopen_call(fn) or _is_httpx_direct_call(fn) or _is_aiohttp_request(fn):
                # Require timeout kw and clamp
                tv = _kw_value(n, "timeout")
                if tv is None:
                    return "network call requires explicit timeout <= 10s"
                v = _const_num(tv)
                if v is not None and v > _MAX_TIMEOUT:
                    return f"timeout too large: {v}s (> {_MAX_TIMEOUT}s)"
                # For requests: forbid verify=False (allow verify=True or default)
                if _is_requests_call(fn) or _is_httpx_direct_call(fn):
                    vv = _kw_value(n, "verify")
                    if isinstance(vv, ast.Constant) and vv.value is False:
                        return "requests.* with verify=False is disallowed"
                # For aiohttp: ssl=False is unsafe
                if _is_aiohttp_request(fn):
                    sv = _kw_value(n, "ssl")
                    if isinstance(sv, ast.Constant) and sv.value is False:
                        return "aiohttp.request with ssl=False is disallowed"
            # Client constructors should also provide timeouts
            if _is_httpx_client_ctor(fn):
                tv = _kw_value(n, "timeout")
                if tv is None:
                    return "httpx.Client requires explicit timeout"
            if _is_aiohttp_session_ctor(fn):
                tv = _kw_value(n, "timeout")
                if tv is None:
                    return "aiohttp.ClientSession requires explicit timeout"
    return None
