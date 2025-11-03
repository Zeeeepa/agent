from __future__ import annotations

import asyncio
import os
import sys
import contextlib
from typing import Tuple, Optional

from jinx.micro.exec.run_exports import write_last_run


async def _run_cmd(*args: str, cwd: Optional[str] = None, timeout_s: Optional[float] = None) -> Tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd or os.getcwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            outs, errs = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            return 124, "", f"timeout after {timeout_s}s"
        code = proc.returncode if proc.returncode is not None else 1
        return code, (outs.decode(errors="ignore") if outs else ""), (errs.decode(errors="ignore") if errs else "")
    except Exception as e:
        return 1, "", f"exec error: {e}"


async def run_tests(pattern: Optional[str] = None, *, chars: Optional[int] = None, timeout_s: Optional[float] = None) -> Tuple[bool, str, str, str]:
    """Run project tests via pytest if available, otherwise unittest discover.

    Returns (ok, stdout, stderr, status). Also persists outputs via run_exports.
    """
    # Pick runner (env override first)
    runner = (os.getenv("JINX_TEST_RUNNER", "auto").strip().lower())
    py = sys.executable or "python"
    cmd: Tuple[str, ...]
    if runner in ("pytest", "auto"):
        # Try pytest
        k_args: Tuple[str, ...] = ("-k", pattern) if pattern else tuple()
        cmd = (py, "-m", "pytest", "-q", *k_args)
        code, out, err = await _run_cmd(*cmd, timeout_s=timeout_s)
        if code in (0, 5):  # 5 = no tests collected
            status = "ok" if code == 0 else "no tests"
            write_last_run(out, None if code == 0 else err or status, None, ok=(code == 0))
            if chars:
                out = out[:max(24, int(chars))]
                err = err[:max(24, int(chars))]
            return (code == 0), out, err, status
        # Fallback to unittest if pytest failed due to import error
    # unittest discover fallback
    start_dir = os.getenv("JINX_TEST_START_DIR", "tests")
    top = os.getenv("JINX_TEST_TOP", None)
    pat = os.getenv("JINX_TEST_PATTERN", "test*.py")
    cmd = (py, "-m", "unittest", "discover", "-v", start_dir, pat)
    code, out, err = await _run_cmd(*cmd, timeout_s=timeout_s)
    status = "ok" if code == 0 else "failure"
    write_last_run(out, None if code == 0 else err or status, None, ok=(code == 0))
    if chars:
        out = out[:max(24, int(chars))]
        err = err[:max(24, int(chars))]
    return (code == 0), out, err, status
