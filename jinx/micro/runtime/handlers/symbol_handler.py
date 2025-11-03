from __future__ import annotations

from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    patch_symbol_python as _patch_symbol,
    should_autocommit as _should_autocommit,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize
from jinx.micro.embeddings.symbol_index import update_symbol_index as _symindex_update
from jinx.async_utils.fs import read_text_raw
from jinx.micro.runtime.patch.write_patch import patch_write as _patch_write
from jinx.micro.runtime.patch.validators_integration import validate_text
from jinx.micro.core.edit_core import finalize_commit
from jinx.micro.common.log import log_info, log_error

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_symbol_patch(tid: str, path: str, symbol: str, replacement: str, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        await report_progress(tid, 15.0, f"preview symbol {symbol} in {path}")
        ok_prev, diff = await _patch_symbol(path, symbol, replacement, preview=True)
        if not ok_prev:
            try:
                log_error("patch.preview.fail", strategy="symbol", path=path, symbol=symbol)
            except Exception:
                pass
            await report_result(tid, False, error=diff)
            return
        try:
            log_info("patch.preview.ok", strategy="symbol", path=path, symbol=symbol)
        except Exception:
            pass
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit("symbol", diff)
        if not okc:
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = "symbol"
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"path": path, "symbol": symbol, "diff": diff})
            return
        await report_progress(tid, 55.0, f"commit symbol {symbol} in {path}")
        # Snapshot current contents for potential revert on validation failure
        try:
            cur_before = await read_text_raw(path)
        except Exception:
            cur_before = ""
        ok_commit, diff2 = await _patch_symbol(path, symbol, replacement, preview=False)
        if ok_commit:
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = "symbol"
            snapshots = {path: cur_before or ""}
            core = await finalize_commit([path], diff2 or "", snapshots=snapshots, strategy="symbol")
            if not core.ok:
                try:
                    log_error("patch.commit.revert", strategy="symbol", path=path, symbol=symbol, errs=len(core.errors))
                except Exception:
                    pass
                await report_result(tid, False, error=(core.errors[0] if core.errors else "validation failed"), result={"path": path, "reverted": True})
                return
            try:
                log_info("patch.commit.ok", strategy="symbol", path=path, symbol=symbol, warns=len(core.warnings or []))
            except Exception:
                pass
            await report_result(tid, True, {"path": path, "symbol": symbol, "diff": diff2, **({"watchdog": core.warnings} if core.warnings else {})})
        else:
            try:
                log_error("patch.commit.fail", strategy="symbol", path=path, symbol=symbol)
            except Exception:
                pass
            await report_result(tid, False, error=diff2)
    except Exception as e:
        try:
            from jinx.micro.common.repair_utils import maybe_schedule_repairs_from_error as _rep
            await _rep(e)
        except Exception:
            pass
        await report_result(tid, False, error=f"symbol patch failed: {e}")
