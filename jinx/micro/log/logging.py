from __future__ import annotations

from jinx.log_paths import INK_SMEARED_DIARY, BLUE_WHISPERS
from jinx.micro.transcript import read_transcript, append_and_trim
from jinx.logger import append_line
from jinx.state import shard_lock


async def glitch_pulse() -> str:
    """Return the current conversation transcript contents.

    Serialized by a shared async lock to avoid interleaved I/O.
    """
    async with shard_lock:
        return await read_transcript(INK_SMEARED_DIARY)


async def blast_mem(x: str, n: int = 500) -> None:
    """Append a line to the transcript, trimming to the last ``n`` lines."""
    async with shard_lock:
        await append_and_trim(INK_SMEARED_DIARY, x, keep_lines=n)


async def bomb_log(t: str, bin: str = BLUE_WHISPERS) -> None:
    """Append a line to a log file (best-effort)."""
    async with shard_lock:
        await append_line(bin, t or "")


__all__ = [
    "glitch_pulse",
    "blast_mem",
    "bomb_log",
]
