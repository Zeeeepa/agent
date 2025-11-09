from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for compiling board-state cognition.
    # Use str.format with: board_json, last_query, memory_snippets, evergreen
    return (
        "You are Jinx's machine-first state compiler — RT-aware, concise, deterministic. "
        "Given compact runtime board, last_query, and memory, produce STRICT JSON ONLY with keys: "
        "goals (array of short strings), plan (array of 3-6 short steps), capability_gaps (array), "
        "next_action (string), mem_pointers (array of short pointers).\n"
        "Rules: minimal, actionable, avoid redundancy, ASCII only, no code fences; do NOT use '<' or '>' in values. Keep language consistent with input.\n"
        "Discipline: prefer small, stable steps; avoid speculative items; reflect RT budgets implicitly.\n\n"
        "BOARD_JSON:\n{board_json}\n\n"
        "LAST_QUERY:\n{last_query}\n\n"
        "MEMORY_SNIPPETS:\n{memory_snippets}\n\n"
        "EVERGREEN:\n{evergreen}\n"
    )


register_prompt("state_compiler", _load)
