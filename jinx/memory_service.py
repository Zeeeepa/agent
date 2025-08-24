from __future__ import annotations


async def optimize_memory(snapshot: str | None = None) -> None:
    """Public facade kept for backward compatibility.

    Delegates to the async memory optimizer worker which ensures FIFO ordering
    and per-turn snapshotting.
    """
    from jinx.memory.optimizer import submit
    await submit(snapshot)
