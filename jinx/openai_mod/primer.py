from __future__ import annotations

from jinx.config import jinx_tag
from jinx.fs_service import wire


async def build_header_and_tag() -> tuple[str, str]:
    """Build instruction header and return it with a code tag identifier.

    Returns (header_plus_prompt, code_tag_id).
    """
    fid, _ = jinx_tag()
    from jinx.config import neon_stat

    chaos = neon_stat()
    header = (
        "\npulse: 1"
        f"\nkey: {fid}"
        f"\nos: {chaos['os']}"
        f"\narch: {chaos['arch']}"
        f"\nhost: {chaos['host']}"
        f"\nuser: {chaos['user']}\n"
    )
    return header + await wire("prompt/burning_logic.txt"), fid
