from __future__ import annotations

# Re-export directly from the local micro-module to avoid circular imports
# when the top-level facade `jinx.net` imports from `jinx.micro.net.client`.
from .client import get_openai_client

__all__ = [
    "get_openai_client",
]
