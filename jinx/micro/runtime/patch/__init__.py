from __future__ import annotations

from .utils import unified_diff, diff_stats, should_autocommit, syntax_check_enabled
from .write_patch import patch_write
from .line_patch import patch_line_range
from .anchor_patch import patch_anchor_insert_after
from .symbol_patch import patch_symbol_python
from .symbol_body_patch import patch_symbol_body_python
from .context_patch import patch_context_replace
from .semantic_patch import patch_semantic_in_file
from .autopatch import AutoPatchArgs, autopatch

__all__ = [
    "unified_diff",
    "diff_stats",
    "should_autocommit",
    "syntax_check_enabled",
    "patch_write",
    "patch_line_range",
    "patch_anchor_insert_after",
    "patch_symbol_python",
    "patch_symbol_body_python",
    "patch_context_replace",
    "patch_semantic_in_file",
    "AutoPatchArgs",
    "autopatch",
]
