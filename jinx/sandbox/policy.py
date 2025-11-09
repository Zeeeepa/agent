from __future__ import annotations

"""
Sandbox policies: compute and apply per-run resource limits.

- On POSIX, use `resource` to clamp CPU time, address space, open files.
- On Windows, gracefully degrade to timeouts only (handled by async_runner).

This module is import-safe on all platforms.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class SandboxLimits:
    cpu_seconds: Optional[int] = None
    as_bytes: Optional[int] = None  # address space / memory cap
    nproc: Optional[int] = None
    nfile: Optional[int] = None


def select_limits(kind: str | None = None) -> SandboxLimits:
    """Select limits by kind or environment defaults.
    kind can be: 'default', 'verify', 'heavy'.
    """
    k = (kind or os.getenv("JINX_SANDBOX_POLICY", "default")).strip().lower()
    def _int(env: str, default: Optional[int]) -> Optional[int]:
        v = os.getenv(env, "")
        if not v:
            return default
        try:
            return int(v)
        except Exception:
            return default
    if k == "verify":
        return SandboxLimits(
            cpu_seconds=_int("JINX_SANDBOX_CPU_S", 5),
            as_bytes=_int("JINX_SANDBOX_AS_BYTES", 256 * 1024 * 1024),
            nfile=_int("JINX_SANDBOX_NOFILE", 256),
        )
    elif k == "heavy":
        return SandboxLimits(
            cpu_seconds=_int("JINX_SANDBOX_CPU_S", 20),
            as_bytes=_int("JINX_SANDBOX_AS_BYTES", 1024 * 1024 * 1024),
            nfile=_int("JINX_SANDBOX_NOFILE", 1024),
        )
    return SandboxLimits(
        cpu_seconds=_int("JINX_SANDBOX_CPU_S", 10),
        as_bytes=_int("JINX_SANDBOX_AS_BYTES", 512 * 1024 * 1024),
        nfile=_int("JINX_SANDBOX_NOFILE", 512),
    )


def apply_limits(lim: SandboxLimits) -> None:
    """Apply limits in current process (POSIX best-effort). No-ops on Windows.
    """
    try:
        import resource  # type: ignore[attr-defined]
    except Exception:
        return
    try:
        if lim.cpu_seconds is not None:
            resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_seconds, lim.cpu_seconds))
    except Exception:
        pass
    try:
        if lim.as_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (lim.as_bytes, lim.as_bytes))
    except Exception:
        pass
    try:
        if lim.nfile is not None:
            resource.setrlimit(resource.RLIMIT_NOFILE, (lim.nfile, lim.nfile))
    except Exception:
        pass
    try:
        if lim.nproc is not None and hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (lim.nproc, lim.nproc))
    except Exception:
        pass
