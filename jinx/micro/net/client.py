from __future__ import annotations

import os
from jinx.bootstrap import ensure_optional, package
import importlib
from typing import Any

openai = ensure_optional(["openai"])["openai"]  # dynamic import


_cortex: Any | None = None


def get_openai_client() -> Any:
    """Return a singleton OpenAI client, honoring optional PROXY env var."""
    global _cortex
    if _cortex is not None:
        return _cortex
    proxy = os.getenv("PROXY")
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
            transport = httpx_socks.SyncProxyTransport.from_url(proxy)
            _cortex = openai.OpenAI(http_client=httpx.Client(transport=transport))
        except Exception:
            # Fallback to direct client if proxy configuration fails
            _cortex = openai.OpenAI()
    else:
        _cortex = openai.OpenAI()
    return _cortex
