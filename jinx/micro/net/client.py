from __future__ import annotations

import os
from urllib.parse import urlparse
from jinx.bootstrap import ensure_optional, package
import importlib
from typing import Any

openai = ensure_optional(["openai"])["openai"]  # dynamic import


_cortex: Any | None = None


def _pick_proxy_env() -> str | None:
    # Preference order: explicit PROXY, then HTTPS_PROXY, then HTTP_PROXY (case-insensitive)
    for key in ("PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.getenv(key)
        if val:
            return val
    return None


def get_openai_client() -> Any:
    """Return a singleton OpenAI client, honoring system proxy env vars.

    Supports both SOCKS and HTTP(S) proxies by constructing an httpx.Client
    with the appropriate transport or proxies mapping.
    """
    global _cortex
    if _cortex is not None:
        return _cortex
    proxy = _pick_proxy_env()
    if proxy:
        try:
            try:
                httpx_socks = importlib.import_module("httpx_socks")
                httpx = importlib.import_module("httpx")
            except ImportError:
                # Ensure both transport and client libraries are present
                package("httpx-socks")
                package("httpx")
                httpx_socks = importlib.import_module("httpx_socks")
                httpx = importlib.import_module("httpx")
            scheme = (urlparse(proxy).scheme or "").lower()
            if scheme.startswith("socks"):
                transport = httpx_socks.SyncProxyTransport.from_url(proxy)
                _cortex = openai.OpenAI(http_client=httpx.Client(transport=transport))
            else:
                # HTTP(S) proxy via httpx native proxies support
                _cortex = openai.OpenAI(http_client=httpx.Client(proxies=proxy))
        except Exception:
            # Fallback to direct client if proxy configuration fails
            _cortex = openai.OpenAI()
    else:
        _cortex = openai.OpenAI()
    return _cortex
