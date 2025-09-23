from __future__ import annotations

"""Network client facade.

Delegates to the micro-module implementation under
`jinx.micro.net.client` while keeping the public API stable.
"""

from jinx.micro.net.client import get_openai_client as get_openai_client


__all__ = [
    "get_openai_client",
]
