from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # One-liner appended to instructions to shape alternative candidate style.
    return (
        "Honor the existing output tag contract. Return a precise, self-contained solution in the required tag format."
    )


register_prompt("consensus_alt", _load)
