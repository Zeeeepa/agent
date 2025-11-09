from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for refining a JSON patch plan to focus on specific files.
    # Use str.format with: goal, top_files_csv, current_plan_json
    return (
        "Revise the JSON plan as Jinx's machine-first internal architect: ruthless, RT-aware.\n"
        "Focus primarily on these files (highest priority first): {top_files_csv}. Do not expand to other files.\n"
        "Keep the exact same JSON schema and constraints (no extra fields). Prefer context/symbol strategies.\n"
        "Constraints: atomic patches, tiny diffs, preserve existing style, ASCII only, no code fences; avoid '<' and '>' in values.\n"
        "- Keep any 'code' field <= 120 lines; prefer smaller when possible.\n"
        "- Minimize imports; avoid adding non-stdlib imports; prefer reusing existing imports.\n"
        "- Allowed strategies reminder: 'symbol','symbol_body','context','line','semantic','write','codemod_rename','codemod_add_import','codemod_replace_import'.\n"
        "- If uncertain, prefer 'semantic' description over risky line ranges.\n"
        "Respect the Risk Policy: avoid denied paths or globs.\n"
        "Goal: {goal}\n"
        "Current plan:\n{current_plan_json}\n"
    )


register_prompt("planner_refine_embed", _load)
