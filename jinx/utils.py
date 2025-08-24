"""Utility helpers.

This module exposes a small set of utilities used across async flows. The
primary export is ``chaos_patch``, an async context manager that patches
``stdout`` to cooperate with ``prompt_toolkit`` rendering. This avoids output
interleaving artifacts when concurrently reading user input and printing logs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
import importlib
from jinx.bootstrap import ensure_optional

# Lazy install of prompt_toolkit, then import patch_stdout from correct submodule
ensure_optional(["prompt_toolkit"])  # installs if missing
patch_stdout = importlib.import_module("prompt_toolkit.patch_stdout").patch_stdout  # type: ignore[assignment]


@asynccontextmanager
async def chaos_patch() -> AsyncIterator[None]:
    """Patch ``stdout`` for clean ``prompt_toolkit`` output in async contexts.

    Yields
    ------
    None
        Control back to the caller with ``stdout`` safely patched.
    """
    with patch_stdout():
        yield
