from __future__ import annotations

from .config import can_render
from typing import Callable


def get_spinner_frames(ascii_only: bool, can: Callable[[str], bool] = can_render) -> str:
    """Return spinner frames as a compact string.
    Preference:
      1) Corner arcs: ◜◝◞◟
      2) ASCII fallback: -|/\
    """
    if ascii_only:
        return "-\\|/"
    return "◜◝◞◟" if all(can(c) for c in "◜◝◞◟") else "-\\|/"
