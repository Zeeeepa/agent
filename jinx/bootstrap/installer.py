from __future__ import annotations

import subprocess
import sys
import time


def package(p: str, *, retries: int = 1, delay_s: float = 1.5) -> None:
    """Install a pip package by name with light retries."""
    attempt = 0
    last_exc: Exception | None = None
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        p,
    ]
    while attempt <= max(0, retries):
        try:
            subprocess.check_call(cmd)
            return
        except Exception as exc:  # pragma: no cover - best-effort bootstrap
            last_exc = exc
            attempt += 1
            if attempt > retries:
                break
            time.sleep(delay_s)
    if last_exc:
        raise last_exc
