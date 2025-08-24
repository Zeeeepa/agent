"""Logging utilities.

Provides lightweight async helpers for appending to transcript and log files
under an async lock to avoid interleaved writes. These helpers deliberately
avoid dependencies on the stdlib ``logging`` module to keep the interactive
runtime straightforward and deterministic.
"""

from __future__ import annotations

from .state import shard_lock
from jinx.transcript import read_transcript, append_and_trim
from jinx.logger import append_line


async def glitch_pulse() -> str:
    """Return the current conversation transcript contents."""
    async with shard_lock:
        return read_transcript("log/soul_fragment.txt")


async def blast_mem(x: str, n: int = 500) -> None:
    """Append a line to the transcript, trimming to the last ``n`` lines.

    Parameters
    ----------
    x : str
        Text to append as a new entry in the transcript.
    n : int, optional
        Maximum number of lines to retain (default 500).
    """
    async with shard_lock:
        append_and_trim("log/soul_fragment.txt", x, keep_lines=n)


async def bomb_log(t: str, bin: str = "log/cortex_wail.txt") -> None:
    """Append a line to a log file.

    Parameters
    ----------
    t : str
        Text to log (``None``-safe).
    bin : str
        Path to the log file (created if missing).
    """
    async with shard_lock:
        append_line(bin, t or "")
