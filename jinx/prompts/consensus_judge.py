from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for LLM judging between two candidates A and B.
    # Use str.format with: a, b
    return (
        "You are a strict code evaluator. Two candidate outputs are provided. "
        "Score which one is better for correctness, structure, and completeness. "
        "Respond with raw JSON only (ASCII only; no code fences, no prose): {\"pick\": \"A|B\", \"score\": 0..1}.\n\n"
        "[A]\n{a}\n\n[B]\n{b}"
    )


register_prompt("consensus_judge", _load)
