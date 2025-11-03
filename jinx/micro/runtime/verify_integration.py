from __future__ import annotations

import os
from typing import Optional, List
from jinx.micro.runtime.contracts import TASK_REQUEST
from jinx.micro.embeddings.search_cache import search_project_cached
from jinx.micro.runtime.static_checks import discover_missing_deps, ensure_requirements_updated
from jinx.micro.runtime.smoke_import import smoke_import_paths
from jinx.logging_service import bomb_log
from jinx.log_paths import BLUE_WHISPERS
from jinx.micro.conversation.cont import load_last_anchors as _load_anchors
from jinx.micro.runtime.patch.import_synth import synthesize_and_patch_imports
from jinx.micro.embeddings.symbol_index import update_symbol_index as _symindex_update
from jinx.micro.common.env import truthy
from jinx.micro.common.log import log_info
from jinx.micro.rt.activity import set_activity as _set_act, set_activity_detail as _set_det


async def last_goal() -> str:
    try:
        anc = await _load_anchors()
    except Exception:
        anc = {}
    try:
        q = (anc.get("questions") or [])
        if q:
            return str(q[-1]).strip()
    except Exception:
        pass
    return ""


async def maybe_verify(goal: Optional[str], files: List[str], diff: str) -> None:
    # Master switch for autorun embedding-based verify
    on = truthy("JINX_VERIFY_AUTORUN", "1")
    # Optional static checks (fast) to discover missing imports and update requirements
    static_on = truthy("JINX_VERIFY_STATIC", "1")
    if static_on and files:
        try:
            missing = await discover_missing_deps(list(files or []))
        except Exception:
            missing = []
        if missing:
            # Best-effort: append to requirements.txt if allowed; log outcome
            try:
                updated, msg = await ensure_requirements_updated(missing)
            except Exception as e:
                updated, msg = False, f"requirements update failed: {e}"
            try:
                await bomb_log(f"[verify.static] missing_deps: {', '.join(missing)}; {msg}", BLUE_WHISPERS)
            except Exception:
                pass
            try:
                log_info("verify.static", missing=len(missing), updated=bool(updated))
            except Exception:
                pass
    # Optional smoke-import to detect immediate import/runtime issues
    smoke_on = truthy("JINX_VERIFY_SMOKE_IMPORT", "1")
    if smoke_on and files:
        try:
            errs = smoke_import_paths(list(files or []))
        except Exception:
            errs = []
        if errs:
            try:
                await bomb_log("\n".join(["[verify.smoke] import checks:"] + errs), BLUE_WHISPERS)
            except Exception:
                pass
            # Attempt targeted repair for missing internal modules detected in errors
            try:
                mods: list[str] = []
                for line in errs:
                    ln = (line or "").strip()
                    # Match forms like: ModuleNotFoundError: No module named 'jinx.foo.bar'
                    if "No module named 'jinx." in ln:
                        try:
                            mod = ln.split("No module named '",1)[1].split("'",1)[0]
                        except Exception:
                            mod = ""
                        if mod and mod.startswith("jinx.") and mod not in mods:
                            mods.append(mod)
                if mods:
                    from jinx.micro.runtime.api import submit_task as _submit  # lazy import
                    try:
                        head = mods[0]
                        more = max(0, len(mods) - 1)
                        label = head if more == 0 else f"{head} (+{more})"
                        _set_act(f"repair: {label}")
                        _set_det({"repair": {"phase": "queued", "mods": mods}})
                    except Exception:
                        pass
                    for m in mods:
                        try:
                            await _submit("repair.import_missing", module=m)
                        except Exception:
                            pass
                    try:
                        log_info("verify.smoke_repair.submitted", count=len(mods))
                    except Exception:
                        pass
            except Exception:
                pass
            # Attempt auto import synthesis on Python files (env-gated)
            synth_on = truthy("JINX_VERIFY_SYNTH_IMPORTS", "1")
            if synth_on:
                changed: list[str] = []
                for p in (files or []):
                    if not str(p).endswith(".py"):
                        continue
                    try:
                        ok, diff = await synthesize_and_patch_imports(p)
                    except Exception:
                        ok, diff = False, ""
                    if ok and diff:
                        changed.append(p)
                if changed:
                    try:
                        await _symindex_update(changed)
                    except Exception:
                        pass
                    # Re-run smoke import after synthesis
                    try:
                        errs2 = smoke_import_paths(list(files or []))
                    except Exception:
                        errs2 = []
                    if errs2:
                        try:
                            await bomb_log("\n".join(["[verify.smoke] after synth:"] + errs2), BLUE_WHISPERS)
                        except Exception:
                            pass
                    try:
                        log_info("verify.smoke_synth", changed=len(changed), errs_after=len(errs2))
                    except Exception:
                        pass
    if not on:
        return
    try:
        await _ensure_verifier()
    except Exception:
        pass
    g = goal or await last_goal()
    if not g:
        return
    try:
        await _verify_embed(g, files=list(files or []), diff=str(diff or ""))
    except Exception:
        pass


# Lazy wrappers to avoid import cycles with runtime.api at module import time
async def _ensure_verifier() -> None:
    try:
        from jinx.micro.verify.verifier import ensure_verifier_running as __ensure
    except Exception:
        return
    await __ensure()  # type: ignore[misc]


async def _verify_embed(goal: str, *, files: List[str], diff: str) -> None:
    try:
        from jinx.micro.verify.verifier import submit_verify_embedding as __submit
    except Exception:
        return
    await __submit(goal, files=files, diff=diff)  # type: ignore[misc]
