from __future__ import annotations

import ast
import asyncio
import difflib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from jinx.async_utils.fs import read_text_raw, write_text
from jinx.micro.embeddings.project_search_api import search_project
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from jinx.micro.runtime.patch import (
    unified_diff as _unified_diff,
    diff_stats as _diff_stats,
    should_autocommit as _should_autocommit,
    patch_write as _patch_write,
    patch_line_range as _patch_line_range,
    patch_anchor_insert_after as _patch_anchor,
    patch_symbol_python as _patch_symbol,
    patch_context_replace as _patch_context,
    patch_symbol_body_python as _patch_symbol_body,
    patch_semantic_in_file as _patch_semantic,
    AutoPatchArgs as _AutoPatchArgs,
    autopatch as _autopatch,
)


def _is_codey(text: str) -> bool:
    return _is_code_like(text or "")


def unified_diff(old: str, new: str, *, path: str = "") -> str:
    """Produce a compact unified diff for preview/logging (delegates to patch.utils)."""
    return _unified_diff(old, new, path=path)


def diff_stats(diff: str) -> Tuple[int, int]:
    """Return (added_lines, removed_lines) ignoring headers (delegates to patch.utils)."""
    return _diff_stats(diff)


def should_autocommit(strategy: str, diff: str) -> Tuple[bool, str]:
    """Delegate to patch.should_autocommit."""
    return _should_autocommit(strategy, diff)


def _syntax_check_enabled() -> bool:
    try:
        return str(os.getenv("JINX_PATCH_CHECK_SYNTAX", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


async def patch_write(path: str, text: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Delegate to patch.patch_write."""
    return await _patch_write(path, text, preview=preview)


async def patch_line_range(path: str, ls: int, le: int, replacement: str, *, preview: bool = False, max_span: Optional[int] = None) -> Tuple[bool, str]:
    """Delegate to patch.patch_line_range."""
    return await _patch_line_range(path, ls, le, replacement, preview=preview, max_span=max_span)


async def patch_context_replace(path: str, before_block: str, replacement: str, *, preview: bool = False, tolerance: float = 0.72) -> Tuple[bool, str]:
    """Delegate to patch.patch_context_replace."""
    return await _patch_context(path, before_block, replacement, preview=preview, tolerance=tolerance)


async def patch_anchor_insert_after(path: str, anchor: str, replacement: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Delegate to patch.patch_anchor_insert_after."""
    return await _patch_anchor(path, anchor, replacement, preview=preview)


async def patch_symbol_python(path: str, symbol: str, replacement: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Delegate to patch.patch_symbol_python."""
    return await _patch_symbol(path, symbol, replacement, preview=preview)


@dataclass
class AutoPatchArgs:
    path: Optional[str] = None
    code: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    symbol: Optional[str] = None
    anchor: Optional[str] = None
    query: Optional[str] = None
    preview: bool = False
    max_span: Optional[int] = None
    force: bool = False
    context_before: Optional[str] = None
    context_tolerance: Optional[float] = None


async def autopatch(args: AutoPatchArgs) -> Tuple[bool, str, str]:
    """Delegate to patch.autopatch, converting to the new AutoPatchArgs type."""
    new_args = _AutoPatchArgs(
        path=args.path,
        code=args.code,
        line_start=args.line_start,
        line_end=args.line_end,
        symbol=args.symbol,
        anchor=args.anchor,
        query=args.query,
        preview=args.preview,
        max_span=args.max_span,
        force=args.force,
        context_before=args.context_before,
        context_tolerance=args.context_tolerance,
    )
    return await _autopatch(new_args)


# Additional improved strategies exposed for convenience
async def patch_symbol_body_python(path: str, symbol: str, body: str, *, preview: bool = False) -> Tuple[bool, str]:
    return await _patch_symbol_body(path, symbol, body, preview=preview)


async def patch_semantic_in_file(path: str, query: str, replacement: str, *, preview: bool = False, topk: Optional[int] = None, margin: Optional[int] = None, tol: Optional[float] = None) -> Tuple[bool, str]:
    return await _patch_semantic(path, query, replacement, preview=preview, topk=topk, margin=margin, tol=tol)


__all__ = [
    "unified_diff",
    "diff_stats",
    "should_autocommit",
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
