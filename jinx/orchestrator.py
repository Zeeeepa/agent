"""Top-level orchestrator.

This module exposes the synchronous ``main()`` function that boots the
asynchronous runtime loop via ``jinx.runtime_service.pulse_core``. Keeping this
adapter minimal ensures clean separation between synchronous CLI entrypoints
and the async runtime core.
"""

from __future__ import annotations

import asyncio
from jinx.bootstrap import load_env, ensure_optional


def main() -> None:
    """Run the async runtime loop and block until completion.

    This function is intentionally synchronous so it can be used directly from
    standard CLI entrypoints without requiring the caller to manage an event
    loop.
    """
    # Ensure environment variables (e.g., OPENAI_API_KEY) are loaded from .env
    load_env()
    # Ensure runtime optional deps are present before importing runtime_service
    ensure_optional([
        "aiofiles",      # async file IO used by runtime
        "regex",         # fuzzy regex stage
        "rapidfuzz",     # fuzzy line matching
        "jedi",          # Python identifier references
        "libcst",        # CST structural patterns
        "astunparse",    # pretty-printing annotations (optional)
    ])

    # Defer import until after dependencies are ensured to avoid early import errors
    from jinx.runtime_service import pulse_core

    asyncio.run(pulse_core())
