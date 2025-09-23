from __future__ import annotations

# Unicode space normalization utilities

_UNICODE_SPACE_MAP = {
    "\u00A0": " ",  # NBSP
    "\u202F": " ",  # NARROW NO-BREAK SPACE
    "\u2007": " ",  # FIGURE SPACE
}


def normalize_unicode_spaces(text: str) -> str:
    t = text
    for k, v in _UNICODE_SPACE_MAP.items():
        t = t.replace(k, v)
    return t
