from __future__ import annotations

from typing import Dict, List, Tuple
import os
import time

from jinx.micro.runtime.watchdog import maybe_warn_filesize
from jinx.async_utils.fs import read_text_raw
from jinx.micro.runtime.patch.validators_integration import validate_text
from jinx.micro.runtime.patch.write_patch import patch_write as _patch_write
from jinx.micro.embeddings.symbol_index import update_symbol_index as _symindex_update
from jinx.micro.runtime.verify_integration import maybe_verify as _maybe_verify
from jinx.micro.common.result import PatchResult
from jinx.micro.common.log import log_info
import os
import py_compile


async def finalize_commit(paths: List[str], diff_text: str, *, snapshots: Dict[str, str] | None = None, strategy: str = "") -> PatchResult:
    """Unified post-commit pipeline.

    - maybe_warn_filesize for changed paths
    - validate via code validators; auto-revert transactionally on violations
    - update symbol index
    - run embedding-based verification (includes static/smoke/synth if enabled)
    """
    changed = [p for p in (paths or []) if p]
    warnings: List[str] = []
    errors: List[str] = []
    t0 = time.perf_counter()

    # 1) Watchdog warnings (size/lines)
    for p in changed:
        try:
            w = await maybe_warn_filesize(p)
        except Exception:
            w = None
        if w:
            warnings.append(w)

    # 2) Validate code; revert on any violation
    t_validate0 = time.perf_counter()
    bad: List[Tuple[str, str]] = []
    for p in changed:
        try:
            cur = await read_text_raw(p)
        except Exception:
            cur = None
        if cur == "":
            # treat empty read as a read error; revert this file
            bad.append((p, "file read error or empty"))
            continue
        ok, msg = await validate_text(p, cur or "")
        if not ok:
            bad.append((p, msg or "validation failed"))

    if bad:
        validate_ms = int((time.perf_counter() - t_validate0) * 1000.0)
        # Transactional revert all changed files to snapshots
        for p in changed:
            try:
                snap = (snapshots or {}).get(p, "")
                await _patch_write(p, snap or "", preview=False)
            except Exception:
                pass
        for p, msg in bad:
            errors.append(f"{p}: {msg}")
        # structured log for revert
        try:
            log_info("finalize_commit.revert", strategy=strategy, files=len(changed), errs=len(bad), validate_ms=validate_ms)
        except Exception:
            pass
        meta = {"strategy": strategy, "validate_ms": validate_ms}
        return PatchResult(False, warnings, errors, meta, paths=changed, diff=diff_text or "", strategy=strategy)

    # 2b) Optional compile check for Python files; revert on any failure
    compile_on = False
    try:
        compile_on = str(os.getenv("JINX_PATCH_IMPORT_CHECK", "0")).strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        compile_on = False
    compile_ms = 0
    if compile_on:
        t_comp0 = time.perf_counter()
        comp_bad: List[Tuple[str, str]] = []
        for p in changed:
            if not p.lower().endswith(".py"):
                continue
            try:
                py_compile.compile(p, doraise=True)
            except Exception as e:
                comp_bad.append((p, f"compile failed: {e}"))
        if comp_bad:
            compile_ms = int((time.perf_counter() - t_comp0) * 1000.0)
            # Revert all changed files
            for p in changed:
                try:
                    snap = (snapshots or {}).get(p, "")
                    await _patch_write(p, snap or "", preview=False)
                except Exception:
                    pass
            for p, msg in comp_bad:
                errors.append(f"{p}: {msg}")
            try:
                log_info("finalize_commit.revert", strategy=strategy, files=len(changed), errs=len(comp_bad), compile_ms=compile_ms)
            except Exception:
                pass
            meta = {"strategy": strategy, "compile_ms": compile_ms}
            return PatchResult(False, warnings, errors, meta, paths=changed, diff=diff_text or "", strategy=strategy)

    # 3) Symbol index update
    t_index0 = time.perf_counter()
    try:
        await _symindex_update(changed)
    except Exception:
        pass
    index_ms = int((time.perf_counter() - t_index0) * 1000.0)

    # 4) Verification (includes static deps + smoke import + synth imports if enabled)
    t_verify0 = time.perf_counter()
    try:
        await _maybe_verify(goal=None, files=changed, diff=diff_text or "")
    except Exception:
        pass
    verify_ms = int((time.perf_counter() - t_verify0) * 1000.0)
    try:
        total_ms = int((time.perf_counter() - t0) * 1000.0)
        log_info("finalize_commit.ok", strategy=strategy, files=len(changed), warns=len(warnings), validate_ms=validate_ms if 'validate_ms' in locals() else 0, compile_ms=compile_ms, index_ms=index_ms, verify_ms=verify_ms, total_ms=total_ms)
    except Exception:
        pass
    meta = {"strategy": strategy, "compile_ms": compile_ms, "index_ms": index_ms, "verify_ms": verify_ms, "total_ms": int((time.perf_counter() - t0) * 1000.0)}
    return PatchResult(True, warnings, errors, meta, paths=changed, diff=diff_text or "", strategy=strategy)
