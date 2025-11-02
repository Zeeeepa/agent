from __future__ import annotations

import os
from typing import List


async def patch_exports_lines() -> List[str]:
    """Return lines that surface recent patch artifacts via export macros.

    These are plain display lines that expand at prompt time.
    """
    return [
        "Recent Patch Preview (may be empty): {{export:last_patch_preview:1}}",
        "Recent Patch Commit (may be empty): {{export:last_patch_commit:1}}",
        "Recent Patch Strategy: {{export:last_patch_strategy:1}}",
        "Recent Patch Reason: {{export:last_patch_reason:1}}",
    ]


async def verify_exports_lines() -> List[str]:
    """Return lines that surface last verification results via export macros."""
    return [
        "Verification Score: {{export:last_verify_score:1}}",
        "Verification Reason: {{export:last_verify_reason:1}}",
        "Verification Files: {{export:last_verify_files:1}}",
    ]


async def run_exports_lines(run_chars: int | None = None) -> List[str]:
    """Return lines that surface last run artifacts (status/stdout/stderr).

    If run_chars is None, uses JINX_MACRO_MEM_PREVIEW_CHARS with sane defaults.
    """
    if run_chars is None:
        try:
            run_chars = max(24, int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
        except Exception:
            run_chars = 160
    return [
        f"Last Run Status: {{{{m:run:status}}}}",
        f"Last Run Stdout: {{{{m:run:stdout:3:chars={run_chars}}}}}",
        f"Last Run Stderr: {{{{m:run:stderr:2:chars={run_chars}}}}}",
    ]


__all__ = [
    "patch_exports_lines",
    "verify_exports_lines",
    "run_exports_lines",
]
