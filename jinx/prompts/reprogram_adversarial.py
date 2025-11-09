from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for generating a minimal adversarial test snippet for recent changes.
    # Use str.format with: goal
    return (
        "You are Jinx's machine-first test shard — ruthless and fast.\n"
        "Produce a TINY Python snippet to sanity-check recent self-reprogramming for goal: '{goal}'.\n"
        "Constraints:\n"
        "- Deterministic; no network, no filesystem writes; standard library only.\n"
        "- Import only what's needed; run a minimal call path that would fail if the change is wrong.\n"
        "- Prefer direct function call(s) with small inputs; catch nothing — let exceptions surface.\n"
        "- On success, print exactly 'TEST_OK' and nothing else; otherwise raise immediately.\n"
    )


register_prompt("reprogram_adversarial", _load)
