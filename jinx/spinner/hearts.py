from __future__ import annotations

from .config import can_render
from typing import Callable, Tuple


def get_hearts(ascii_only: bool, can: Callable[[str], bool] = can_render) -> Tuple[str, str]:
    """Return (heart_a, heart_b) for pulse.
    Preference: ❤ ↔ ♡, fallback to <3.
    If only one of them is renderable, use that for both to avoid tofu.
    """
    if ascii_only:
        return "<3", "<3"

    full, hollow = "❤", "♡"
    full_ok = can(full)
    hollow_ok = can(hollow)

    if full_ok and hollow_ok:
        return full, hollow

    if full_ok or hollow_ok:
        single = full if full_ok else hollow
        return single, single

    return "<3", "<3"
