from __future__ import annotations

import asyncio
import os
import sys
import venv
from typing import Optional


async def create_venv(stage_dir: str) -> Optional[str]:
    """Create a virtualenv under stage_dir/.venv and return python path.
    Never raises; returns None on failure.
    """
    try:
        vdir = os.path.join(stage_dir, ".venv")
        builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=False, upgrade=False)
        await asyncio.to_thread(builder.create, vdir)
        py = os.path.join(vdir, "Scripts", "python.exe") if os.name == "nt" else os.path.join(vdir, "bin", "python")
        if not os.path.isfile(py):
            return None
        return py
    except Exception:
        return None


async def install_requirements(python_exe: str, stage_dir: str, *, timeout_s: float = 12.0) -> bool:
    """Install requirements if requirements.txt exists in stage.
    Returns True on success or if no requirements file; False on failure.
    """
    req = None
    try:
        for name in ("requirements.txt", "req.txt"):
            cand = os.path.join(stage_dir, name)
            if os.path.isfile(cand):
                req = cand
                break
    except Exception:
        req = None
    if not req:
        return True
    try:
        proc = await asyncio.create_subprocess_exec(
            python_exe, "-m", "pip", "install", "-r", req,
            cwd=stage_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return False
        return proc.returncode == 0
    except Exception:
        return False
