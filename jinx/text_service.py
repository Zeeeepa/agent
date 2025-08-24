"""Text utilities."""

from __future__ import annotations


def slice_fuse(x: str, lim: int = 100_000) -> str:
    """Symmetrically truncate long text with a center ellipsis tag.

    Parameters
    ----------
    x : str
        Text to potentially truncate.
    lim : int
        Maximum resulting length.
    """
    if len(x) <= lim:
        return x
    tag = f"\n...[truncated {len(x)-lim} chars]...\n"
    half = (lim - len(tag)) // 2
    return x[:half] + tag + x[-half:]
