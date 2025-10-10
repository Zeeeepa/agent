from __future__ import annotations

import os
from typing import Optional

from jinx.micro.llm.prompt_filters import sanitize_prompt_for_external_api as _sanitize

async def log_debug(feature: str, line: str) -> None:
    """Append a short debug line to Blue Whispers if debugging is enabled.

    Controlled by env:
    - JINX_CONV_DEBUG=1 enables logging
    - JINX_CONV_DEBUG_FEATURES can filter features (comma-separated)
    """
    try:
        on = str(os.getenv("JINX_CONV_DEBUG", "0")).lower() in ("1","true","on","yes")
        if not on:
            return
        allow = str(os.getenv("JINX_CONV_DEBUG_FEATURES", "")).strip().lower()
        if allow:
            feats = {p.strip() for p in allow.split(',') if p.strip()}
            if feature not in feats:
                return
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        await _append(BLUE_WHISPERS, f"[conv:{feature}] {_sanitize(line)[:400]}")
    except Exception:
        return

__all__ = ["log_debug"]
