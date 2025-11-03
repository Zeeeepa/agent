from __future__ import annotations

from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    patch_anchor_insert_after as _patch_anchor,
    should_autocommit as _should_autocommit,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize
from jinx.async_utils.fs import read_text_raw
from jinx.micro.core.edit_core import finalize_commit

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_anchor_patch(tid: str, path: str, anchor: str, replacement: str, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        await report_progress(tid, 15.0, f"preview anchor '{anchor}' in {path}")
        ok_prev, diff = await _patch_anchor(path, anchor, replacement, preview=True)
        if not ok_prev:
            await report_result(tid, False, error=diff)
            return
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit("anchor", diff)
        if not okc:
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = "anchor"
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"path": path, "anchor": anchor, "diff": diff})
            return
        await report_progress(tid, 55.0, f"commit anchor '{anchor}' in {path}")
        # snapshot for transactional revert
        try:
            cur_before = await read_text_raw(path)
        except Exception:
            cur_before = ""
        ok_commit, diff2 = await _patch_anchor(path, anchor, replacement, preview=False)
        if ok_commit:
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = "anchor"
            core = await finalize_commit([path], diff2 or "", snapshots={path: cur_before or ""}, strategy="anchor")
            if not core.ok:
                await report_result(tid, False, error=(core.errors[0] if core.errors else "validation failed"), result={"path": path, "reverted": True})
                return
            await report_result(tid, True, {"path": path, "anchor": anchor, "diff": diff2, **({"watchdog": core.warnings} if core.warnings else {})})
        else:
            await report_result(tid, False, error=diff2)
    except Exception as e:
        await report_result(tid, False, error=f"anchor patch failed: {e}")
