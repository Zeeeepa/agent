from __future__ import annotations

from typing import Optional, Tuple


def build_chains(synth: str, err: Optional[str]) -> Tuple[str, int]:
    """Combine current transcript with optional recent error.

    Returns (chains, decay_points). Decay points is 50 when a new error line
    is appended, otherwise 0.
    """
    s = synth.strip()
    if err and err.strip() not in s:
        return s + "\n" + err.strip(), 50
    return s, 0
