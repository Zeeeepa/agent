from __future__ import annotations

from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    AutoPatchArgs,
    autopatch as _autopatch,
    should_autocommit as _should_autocommit,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_auto_patch(tid: str, a: AutoPatchArgs, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        await report_progress(tid, 12.0, "auto preview")
        a.preview = True
        ok_prev, strat, diff = await _autopatch(a)
        if not ok_prev:
            await report_result(tid, False, error=f"{strat}: {diff}")
            return
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit(strat, diff)
        if not okc and not a.force:
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = str(strat)
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"strategy": strat, "diff": diff})
            return
        await report_progress(tid, 55.0, "auto commit")
        a.preview = False
        ok_commit, strat2, diff2 = await _autopatch(a)
        if ok_commit:
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = str(strat2)
            warn = None
            try:
                if a.path:
                    warn = await maybe_warn_filesize(a.path)
                    if warn:
                        exports["last_watchdog_warn"] = warn
            except Exception:
                warn = None
            await report_result(tid, True, {"strategy": strat2, "diff": diff2, **({"watchdog": warn} if warn else {})})
            try:
                files = [a.path] if a.path else []
            except Exception:
                files = []
            await verify_cb(goal=None, files=files, diff=diff2)
        else:
            await report_result(tid, False, error=f"{strat2}: {diff2}")
    except Exception as e:
        await report_result(tid, False, error=f"auto patch failed: {e}")
