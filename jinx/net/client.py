from __future__ import annotations

import os
from jinx.bootstrap import ensure_optional, package

openai = ensure_optional(["openai"])["openai"]  # type: ignore


_cortex: openai.OpenAI | None = None


def get_openai_client() -> openai.OpenAI:
    """Return a singleton OpenAI client, honoring optional PROXY env var."""
    global _cortex
    if _cortex is not None:
        return _cortex
    proxy = os.getenv("PROXY")
    if proxy:
        try:
            from httpx_socks import SyncProxyTransport
            import httpx
        except ImportError:
            # Ensure both transport and client libraries are present
            package("httpx-socks")
            package("httpx")
            from httpx_socks import SyncProxyTransport
            import httpx
        _cortex = openai.OpenAI(http_client=httpx.Client(transport=SyncProxyTransport.from_url(proxy)))
    else:
        _cortex = openai.OpenAI()
    return _cortex
