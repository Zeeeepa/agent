from __future__ import annotations

from typing import Any, Dict, List, Callable, Awaitable

from jinx.micro.runtime.patch import AutoPatchArgs
from jinx.micro.runtime.handlers import (
    handle_write as handle_write,
    handle_line_patch as handle_line_patch,
    handle_symbol_patch as handle_symbol_patch,
    handle_anchor_patch as handle_anchor_patch,
    handle_auto_patch as handle_auto_patch,
    handle_batch_patch as handle_batch_patch,
    handle_dump_symbol as handle_dump_symbol,
    handle_dump_by_query as handle_dump_by_query,
    handle_dump_by_query_global as handle_dump_by_query_global,
    handle_refactor_move_symbol as handle_refactor_move_symbol,
    handle_refactor_split_file as handle_refactor_split_file,
)

# Thin delegator for backward compatibility. All logic now lives under
# jinx/micro/runtime/handlers/ micro-modules.

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]

__all__ = [
    "AutoPatchArgs",
    "handle_write",
    "handle_line_patch",
    "handle_symbol_patch",
    "handle_anchor_patch",
    "handle_auto_patch",
    "handle_batch_patch",
    "handle_dump_symbol",
    "handle_dump_by_query",
    "handle_dump_by_query_global",
    "handle_refactor_move_symbol",
    "handle_refactor_split_file",
]
