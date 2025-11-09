from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for generating a minimal, safe Python module for a missing import.
    # Use str.format with: module
    return (
        "Generate a minimal, SAFE Python module implementation for '{module}'.\n"
        "Constraints:\n"
        "- No external dependencies; avoid heavy logic;\n"
        "- Provide only minimal classes/functions likely to be imported;\n"
        "- Never raise on import; safe defaults; export dummies if necessary;\n"
        "- Output ONLY valid Python code, no explanations or fences.\n"
    )


register_prompt("repair_stub", _load)
