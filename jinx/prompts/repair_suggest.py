from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for generating a minimal repair suggestion from error context.
    # Use str.format with: error_type, error_message, file_path, line_number,
    # error_line, context_before_text, context_after_text, containing_scope
    return (
        "You are Jinx's machine-first repair agent — ruthless, precise, RT-aware.\n"
        "Analyze and fix this Python error with the smallest safe change.\n\n"
        "ERROR: {error_type}: {error_message}\n\n"
        "FILE: {file_path}\n"
        "LINE: {line_number}\n\n"
        "ERROR LINE:\n"
        "{error_line}\n\n"
        "CONTEXT BEFORE:\n"
        "{context_before_text}\n\n"
        "CONTEXT AFTER:\n"
        "{context_after_text}\n\n"
        "CONTAINING SCOPE: {containing_scope}\n\n"
        "Constraints:\n"
        "- Minimal, atomic diff; preserve style; avoid broad refactors.\n"
        "- RT-conscious: avoid heavy dependencies or blocking patterns.\n"
        "- ASCII only; no code fences; no triple quotes anywhere; do NOT use try/except.\n"
        "- Keep fix code small (≤120 lines) and focused on the failing region.\n\n"
        "Provide:\n"
        "1. Root cause analysis (concise)\n"
        "2. Minimal fix (only change what's necessary) — show final corrected code for the target region\n"
        "3. Confidence level (0.0-1.0)\n"
        "4. Estimated impact (low/medium/high/critical)\n\n"
        "Format response as:\n"
        "<analysis>...</analysis>\n"
        "<fix>\n[fixed code here]\n</fix>\n"
        "<confidence>0.X</confidence>\n"
        "<impact>level</impact>\n"
    )


register_prompt("repair_suggest", _load)
