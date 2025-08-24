from __future__ import annotations

from jinx.async_utils.fs import read_text_raw, write_text
from jinx.state import shard_lock
from jinx.log_paths import INK_SMEARED_DIARY, EVERGREEN_MEMORY


async def read_evergreen() -> str:
    async with shard_lock:
        evergreen = await read_text_raw(EVERGREEN_MEMORY)
    return evergreen or ""


def ensure_nl(s: str) -> str:
    return s + ("\n" if s and not s.endswith("\n") else "")


async def write_state(compact: str, durable: str | None) -> None:
    compact_out = ensure_nl(compact)
    async with shard_lock:
        await write_text(INK_SMEARED_DIARY, compact_out)
        if durable is not None:
            durable_out = ensure_nl(durable)
            await write_text(EVERGREEN_MEMORY, durable_out)
