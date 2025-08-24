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
    full = "❤" if can("❤") else None
    hollow = "♡" if can("♡") else None
    if full and hollow:
        return full, hollow
    if full:
        return full, full
    if hollow:
        return hollow, hollow
    return "<3", "<3"
