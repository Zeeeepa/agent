from __future__ import annotations

import re
from typing import Callable, Awaitable, List, Dict, Optional

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    patch_write as _patch_write,
    should_autocommit as _should_autocommit,
    diff_stats as _diff_stats,
)
from jinx.async_utils.fs import read_text_raw
from jinx.micro.core.edit_core import finalize_commit
from jinx.micro.common.log import log_info, log_error
from jinx.micro.embeddings.symbol_index import update_symbol_index as _symindex_update
import ast

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


def _compile_flags(flags: Optional[str]) -> int:
    f = (flags or "").lower()
    out = 0
    if "i" in f:
        out |= re.IGNORECASE
    if "m" in f:
        out |= re.MULTILINE
    if "s" in f:
        out |= re.DOTALL
    if "x" in f:
        out |= re.VERBOSE
    return out


async def handle_regex_patch(tid: str, path: str, pattern: str, replacement: str, *, flags: Optional[str], verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        # Read current content
        try:
            original = await read_text_raw(path)
        except Exception:
            await report_result(tid, False, error=f"cannot read: {path}")
            return
        try:
            rx = re.compile(pattern, _compile_flags(flags))
        except re.error as e:
            await report_result(tid, False, error=f"invalid regex: {e}")
            return
        new_text, count = rx.subn(replacement, original)
        if count <= 0 or new_text == original:
            await report_result(tid, False, error="no match or no change")
            return
        await report_progress(tid, 14.0, f"preview regex replace matches={count}")
        ok_prev, diff = await _patch_write(path, new_text, preview=True)
        if not ok_prev:
            try:
                log_error("patch.preview.fail", strategy="regex", path=path, matches=count)
            except Exception:
                pass
            await report_result(tid, False, error=diff)
            return
        try:
            add_p, rem_p = _diff_stats(diff)
            log_info("patch.preview.ok", strategy="regex", path=path, matches=count, add=add_p, rem=rem_p)
        except Exception:
            pass
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit("regex", diff)
        if not okc:
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = "regex"
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"path": path, "diff": diff, "matches": count})
            return
        await report_progress(tid, 45.0, f"commit regex replace matches={count}")
        # Commit via write patcher and validate
        # Pre-commit AST check for Python files to avoid committing syntax errors
        try:
            if path.lower().endswith(".py"):
                ast.parse(new_text)
        except SyntaxError as se:
            await report_result(tid, False, error=f"syntax error: {se}")
            return
        ok_commit, diff2 = await _patch_write(path, new_text, preview=False)
        if ok_commit:
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = "regex"
            core = await finalize_commit([path], diff2 or "", snapshots={path: original or ""}, strategy="regex")
            if not core.ok:
                try:
                    log_error("patch.commit.revert", strategy="regex", path=path, errs=len(core.errors))
                except Exception:
                    pass
                await report_result(tid, False, error=(core.errors[0] if core.errors else "validation failed"), result={"path": path, "reverted": True})
                return
            try:
                log_info("patch.commit.ok", strategy="regex", path=path, matches=count, warns=len(core.warnings or []))
            except Exception:
                pass
            # Update symbol index (best-effort)
            try:
                await _symindex_update([path])
            except Exception:
                pass
            await report_result(tid, True, {"path": path, "diff": diff2, "matches": count, **({"watchdog": core.warnings} if core.warnings else {})})
        else:
            try:
                log_error("patch.commit.fail", strategy="regex", path=path, matches=count)
            except Exception:
                pass
            await report_result(tid, False, error=diff2)
    except Exception as e:
        try:
            from jinx.micro.common.repair_utils import maybe_schedule_repairs_from_error as _rep
            await _rep(e)
        except Exception:
            pass
        await report_result(tid, False, error=f"regex patch failed: {e}")
