from __future__ import annotations

import os
from typing import Any


_FALSE = {"", "0", "false", "off", "no"}


def _is_on(val: str | None, default: str = "1") -> bool:
    v = (val if val is not None else default)
    return (v.strip().lower() not in _FALSE)


def _set_default(name: str, value: str) -> None:
    if os.getenv(name) in (None, ""):
        os.environ[name] = value


def apply_auto_defaults(settings: Any | None = None) -> None:
    """Apply autonomous defaults so the user doesn't need manual config.

    Only sets variables when they are not already defined by the environment.
    """
    # Global auto-mode: if user didn't explicitly turn it off, treat as ON
    auto_on = _is_on(os.getenv("JINX_AUTO_MODE"), default="1")
    if not auto_on:
        return

    # Spinner always visible, toolbar mode, gentle redraw
    _set_default("JINX_SPINNER_ENABLE", "1")
    _set_default("JINX_SPINNER_MODE", "toolbar")
    _set_default("JINX_SPINNER_MIN_UPDATE_MS", "160")
    _set_default("JINX_SPINNER_REDRAW_ONLY_ON_CHANGE", "1")

    # Retrieval + memory context + brain enabled
    _set_default("EMBED_PROJECT_ENABLE", "1")
    _set_default("EMBED_BRAIN_ENABLE", "1")

    # Include memory-backed context block during orchestration
    _set_default("JINX_EMBED_MEMORY_CTX", "1")

    # Planner-generated guidance/context: on by default but small budgets
    _set_default("JINX_PLANNER_CTX", "1")
    _set_default("JINX_CHAINED_DIALOG_CTX_MS", "140")
    _set_default("JINX_CHAINED_PROJECT_CTX_MS", "500")

    # Keep unified context compact and well-spaced
    _set_default("JINX_CTX_COMPACT_ORCH", "1")

    # Streaming fast-path ON
    _set_default("JINX_LLM_STREAM_FASTPATH", "1")

    # Priority dispatcher ON by default
    _set_default("JINX_RUNTIME_USE_PRIORITY_QUEUE", "1")

    # Conservative evergreen by default (can be enabled later by logic)
    _set_default("JINX_EVERGREEN_SEND", "0")

    # Reasonable concurrency defaults
    _set_default("JINX_FRAME_MAX_CONC", "2")

    # Tight but safe RT budgets for staging where applicable
    _set_default("EMBED_SLICE_MS", "12")

    # Locator fast-lane and classifier defaults (autonomous)
    _set_default("JINX_SIMPLE_LOCATOR_SCAN", "3")
    _set_default("JINX_LOCATOR_THRESH", "0.06")
    _set_default("JINX_LOCATOR_VEC_MS", "120")
    _set_default("JINX_LOCATOR_VEC_K", "6")
    _set_default("JINX_LOCATOR_CACHE_TTL", "60")

    # Intake multi-split defaults
    _set_default("JINX_MULTI_SPLIT_ENABLE", "1")
    _set_default("JINX_MULTI_SPLIT_MAX", "6")

    # Scheduler tuning defaults
    _set_default("JINX_GROUP_MAX_CONC", "2")
    _set_default("JINX_FRAME_DRAIN_MAX", "16")

    # Prefetch broker concurrency
    _set_default("JINX_PREFETCH_BROKER_CONC", "3")

    # Embed observer small concurrency
    _set_default("JINX_EMBEDOBS_CONC", "2")

