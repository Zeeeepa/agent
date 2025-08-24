"""Network client bootstrap and proxy support for OpenAI SDK."""

from __future__ import annotations

from jinx.net import get_openai_client


def get_cortex():
    """Return a singleton OpenAI client (delegates to micro-module)."""
    return get_openai_client()
