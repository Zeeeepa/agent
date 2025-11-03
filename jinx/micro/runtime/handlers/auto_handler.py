from __future__ import annotations

from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    AutoPatchArgs,
    autopatch as _autopatch,
    should_autocommit as _should_autocommit,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize
from jinx.async_utils.fs import read_text_raw
from jinx.micro.runtime.patch.write_patch import patch_write as _patch_write
from jinx.micro.runtime.patch.validators_integration import validate_text
from jinx.micro.core.edit_core import finalize_commit
from jinx.micro.common.log import log_info, log_error

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_auto_patch(tid: str, a: AutoPatchArgs, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        await report_progress(tid, 12.0, "auto preview")
        a.preview = True
        ok_prev, strat, diff = await _autopatch(a)
        if not ok_prev:
            try:
                log_error("patch.preview.fail", strategy=str(strat), path=a.path or "", symbol=a.symbol or "", anchor=a.anchor or "", line_start=a.line_start or 0, line_end=a.line_end or 0)
            except Exception:
                pass
            await report_result(tid, False, error=f"{strat}: {diff}")
            return
        try:
            log_info("patch.preview.ok", strategy=str(strat), path=a.path or "")
        except Exception:
            pass
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit(strat, diff)
        if not okc and not a.force:
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = str(strat)
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"strategy": strat, "diff": diff})
            return
        await report_progress(tid, 55.0, "auto commit")
        # Snapshot current contents to allow revert on validation failure when path known
        cur_before = ""
        try:
            if a.path:
                cur_before = await read_text_raw(a.path)
        except Exception:
            cur_before = ""
        a.preview = False
        ok_commit, strat2, diff2 = await _autopatch(a)
        if ok_commit:
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = str(strat2)
            # Use core finalize when path known; otherwise keep minimal success path
            if a.path:
                snapshots = {a.path: cur_before or ""}
                core = await finalize_commit([a.path], diff2 or "", snapshots=snapshots, strategy=str(strat2))
                if not core.ok:
                    try:
                        log_error("patch.commit.revert", strategy=str(strat2), path=a.path, errs=len(core.errors))
                    except Exception:
                        pass
                    await report_result(tid, False, error=(core.errors[0] if core.errors else "validation failed"), result={"strategy": strat2, "reverted": True})
                    return
                try:
                    log_info("patch.commit.ok", strategy=str(strat2), path=a.path, warns=len(core.warnings or []))
                except Exception:
                    pass
                await report_result(tid, True, {"strategy": strat2, "diff": diff2, **({"watchdog": core.warnings} if core.warnings else {})})
            else:
                # Unknown path: report basic success; downstream verify may still run elsewhere
                await report_result(tid, True, {"strategy": strat2, "diff": diff2})
        else:
            try:
                log_error("patch.commit.fail", strategy=str(strat2), path=a.path or "")
            except Exception:
                pass
            await report_result(tid, False, error=f"{strat2}: {diff2}")
    except Exception as e:
        try:
            from jinx.micro.common.repair_utils import maybe_schedule_repairs_from_error as _rep
            await _rep(e)
        except Exception:
            pass
        await report_result(tid, False, error=f"auto patch failed: {e}")
