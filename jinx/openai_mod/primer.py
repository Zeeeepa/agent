from __future__ import annotations

from jinx.config import jinx_tag
from jinx.prompts import get_prompt


async def build_header_and_tag(prompt_override: str | None = None) -> tuple[str, str]:
    """Build instruction header and return it with a code tag identifier.

    Returns (header_plus_prompt, code_tag_id).
    If ``prompt_override`` is provided, that prompt name is used instead of the global default.
    """
    fid, _ = jinx_tag()
    from jinx.config import neon_stat, PROMPT_NAME

    chaos = neon_stat()
    header = (
        "\npulse: 1"
        f"\nkey: {fid}"
        f"\nos: {chaos['os']}"
        f"\narch: {chaos['arch']}"
        f"\nhost: {chaos['host']}"
        f"\nuser: {chaos['user']}\n"
    )
    # Fill template variables like {key} inside the prompt text
    active_name = (prompt_override or PROMPT_NAME)
    prompt = get_prompt(active_name).format(key=fid)
    return header + prompt, fid
