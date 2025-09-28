from __future__ import annotations

from jinx.micro.llm.chain_utils import truthy_env
from jinx.micro.llm.chain_persist import persist_brain
from jinx.logger.file_logger import append_line as _log_append
from jinx.log_paths import BLUE_WHISPERS


async def finalize_context(user_text: str, plan: dict, final_ctx: str) -> None:
    if not final_ctx:
        return
    # Optional persist brain snapshot for embeddings ingestion
    try:
        if truthy_env("JINX_CHAINED_PERSIST_BRAIN", "1"):
            await persist_brain(user_text, plan, final_ctx)
    except Exception:
        pass
    # Optional developer echo for inspection in logs
    try:
        if truthy_env("JINX_CHAINED_DEV_ECHO", "0"):
            preview = final_ctx if len(final_ctx) <= 4000 else (final_ctx[:4000] + "\n...<truncated>")
            await _log_append(BLUE_WHISPERS, f"[CHAIN]\n{preview}")
    except Exception:
        pass
