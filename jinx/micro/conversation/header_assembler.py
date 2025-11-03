from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from jinx.conversation.formatting import build_header as _build_header, ensure_header_block_separation as _ensure_sep


@dataclass
class HeaderInput:
    base_ctx: str
    proj_ctx: str
    plan_ctx: str
    cont_block: str
    turns_block: str
    memsel_block: str
    memory_text: str
    evergreen_text: str
    task_text: str
    error_text: Optional[str]


def assemble_context_payload(inp: HeaderInput) -> str:
    """Assemble the multi-block context payload to be sent to the LLM before header.

    Order: base_ctx, proj_ctx, plan_ctx, cont_block, turns_block, memsel_block.
    """
    parts = [
        inp.base_ctx or "",
        inp.proj_ctx or "",
        inp.plan_ctx or "",
        inp.cont_block or "",
        inp.turns_block or "",
        inp.memsel_block or "",
    ]
    return "\n".join([p for p in parts if p])


def assemble_header(ctx_body: str, inp: HeaderInput) -> Tuple[str, str]:
    """Build the <header> section and return (header_text, chains_with_header).

    The second return value is the normalized chains string that includes the header
    at the top and has proper separation enforced.
    """
    header_text = _build_header(
        ctx_body,
        inp.memory_text or "",
        inp.task_text or "",
        (inp.error_text.strip() if inp.error_text else None),
        inp.evergreen_text or "",
    )
    chains = header_text if header_text else ""
    # Ensure separation so chain blocks do not collide with header
    chains = _ensure_sep(chains)
    return header_text, chains
