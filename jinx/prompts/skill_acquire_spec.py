from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for skill acquisition JSON {path, code}.
    # Use str.format with: query, suggested_path
    return (
        "You are Jinx's machine-first skill engineer — ruthless minimalism, RT-aware.\n"
        "Task: generate a minimal Python skill module that directly helps answer the user query.\n"
        "Return STRICT JSON ONLY with keys: path (string), code (string). No code fences. ASCII only.\n"
        "Constraints:\n"
        "- Micro-modular layout; RT-friendly; avoid blocking IO at import time.\n"
        "- Expose 'async def handle(query: str) -> str' as the entrypoint.\n"
        "- No external dependencies beyond the Python standard library.\n"
        "- Deterministic, concise, and safe; keep code small and focused.\n"
        "- ASCII only; no triple quotes anywhere; do NOT use try/except.\n"
        "- No network or filesystem writes at import or top-level; avoid global side effects.\n"
        "User query: {query}\n"
        "Suggested path (under jinx/skills): {suggested_path}\n"
    )


register_prompt("skill_acquire_spec", _load)
