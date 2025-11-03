from __future__ import annotations

import os
import re
from typing import Callable, Awaitable, List, Dict, Iterable, Optional
from fnmatch import fnmatch

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    patch_write as _patch_write,
    should_autocommit as _should_autocommit,
    diff_stats as _diff_stats,
)
from jinx.async_utils.fs import read_text_raw
from jinx.micro.core.edit_core import finalize_commit
from jinx.micro.common.log import log_info, log_error

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]

_DEFAULT_EXCLUDE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules", "emb", "build", "dist"}


def _iter_files(root: str, includes: Optional[List[str]], excludes: Optional[List[str]], exts: Optional[List[str]]) -> Iterable[str]:
    root = os.path.abspath(root or ".")
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in _DEFAULT_EXCLUDE_DIRS and not d.startswith(".")]
        for fn in filenames:
            ap = os.path.join(dirpath, fn)
            rel = os.path.relpath(ap, start=root).replace("\\", "/")
            if exts:
                ok_ext = any(rel.lower().endswith("." + e.lower().lstrip(".")) for e in exts)
                if not ok_ext:
                    continue
            ok = True
            if includes:
                ok = any(fnmatch(rel, pat) for pat in includes)
            if ok and excludes:
                if any(fnmatch(rel, pat) for pat in excludes):
                    ok = False
            if ok:
                yield ap


def _compile_flags(flags: Optional[str]) -> int:
    f = (flags or "").lower()
    out = 0
    if "i" in f: out |= re.IGNORECASE
    if "m" in f: out |= re.MULTILINE
    if "s" in f: out |= re.DOTALL
    if "x" in f: out |= re.VERBOSE
    return out


async def handle_find_replace(
    tid: str,
    *,
    root: str,
    pattern: str,
    replacement: str,
    verify_cb: VerifyCB,
    exports: Dict[str, str],
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    exts: Optional[List[str]] = None,
    flags: Optional[str] = None,
    max_files: Optional[int] = None,
) -> None:
    """Multi-file regex search/replace with preview/commit and unified validation.

    Args:
        root: root directory to scan
        pattern: regex pattern (Python syntax)
        replacement: replacement string
        includes: optional list of glob patterns relative to root
        excludes: optional list of glob patterns relative to root
        exts: optional list of file extensions (e.g., ["py","md"]) to include
        flags: optional regex flags (e.g., "im")
        max_files: optional cap on number of files to modify
    """
    try:
        try:
            rx = re.compile(pattern, _compile_flags(flags))
        except re.error as e:
            await report_result(tid, False, error=f"invalid regex: {e}")
            return
        # Preview phase: collect diffs
        previews: List[Dict[str, str | int | bool]] = []
        combined_diff_parts: List[str] = []
        changed_files: List[str] = []
        originals: Dict[str, str] = {}
        count_total = 0
        files_seen = 0
        await report_progress(tid, 8.0, "scan files")
        for ap in _iter_files(root, includes, excludes, exts):
            if max_files is not None and len(changed_files) >= max_files:
                break
            files_seen += 1
            try:
                text = await read_text_raw(ap)
            except Exception:
                continue
            new_text, count = rx.subn(replacement, text)
            if count > 0 and new_text != text:
                ok_prev, diff = await _patch_write(ap, new_text, preview=True)
                previews.append({"path": ap, "ok": ok_prev, "matches": count})
                if ok_prev:
                    combined_diff_parts.append(diff)
                    changed_files.append(ap)
                    originals[ap] = text
                    count_total += count
        combined_diff = "\n".join([d for d in combined_diff_parts if d])
        add, rem = _diff_stats(combined_diff)
        exports["last_patch_preview"] = combined_diff or ""
        exports["last_patch_strategy"] = "find_replace"
        await report_progress(tid, 22.0, f"preview {len(changed_files)} files, matches={count_total}")
        okc, reason = _should_autocommit("find_replace", combined_diff)
        if not okc:
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"files": len(changed_files), "matches": count_total, "diff_add": add, "diff_rem": rem})
            return
        # Commit phase
        await report_progress(tid, 55.0, f"commit {len(changed_files)} files")
        combined_commit_parts: List[str] = []
        for ap in changed_files:
            # Use previously computed replacement text to avoid re-running heavy RX; recompute defensively
            try:
                text = originals.get(ap) or await read_text_raw(ap)
            except Exception:
                continue
            new_text, count = rx.subn(replacement, text)
            ok_commit, diff2 = await _patch_write(ap, new_text, preview=False)
            if ok_commit and diff2:
                combined_commit_parts.append(diff2)
        combined_commit = "\n".join([d for d in combined_commit_parts if d])
        # Unified finalize
        files_norm = list({p.replace('\\','/') for p in changed_files if p})
        core = await finalize_commit(files_norm, combined_commit or "", snapshots=originals, strategy="find_replace")
        if not core.ok:
            try:
                log_error("find_replace.commit.revert", files=len(files_norm), errs=len(core.errors))
            except Exception:
                pass
            await report_result(tid, False, error=(core.errors[0] if core.errors else "validation failed"), result={"errors": core.errors, "reverted_files": files_norm})
            return
        try:
            log_info("find_replace.commit.ok", files=len(files_norm), add=add, rem=rem, warns=len(core.warnings or []))
        except Exception:
            pass
        await report_result(tid, True, {"files": len(files_norm), "matches": count_total, "diff_add": add, "diff_rem": rem, **({"watchdog": core.warnings} if core.warnings else {})})
    except Exception as e:
        await report_result(tid, False, error=f"find_replace failed: {e}")
