from __future__ import annotations

import asyncio
import os
from typing import Optional, Tuple

try:
    from jinx.rt.admission import guard as _guard
except Exception:  # pragma: no cover
    _guard = None  # type: ignore

try:
    from jinx.micro.embeddings.project_pipeline import embed_file as _embed_file
    from jinx.micro.embeddings.project_util import sha256_path as _sha_path
except Exception:  # pragma: no cover
    _embed_file = None  # type: ignore
    _sha_path = None  # type: ignore


# Module-level minimal queue with lazy workers
_Q: Optional[asyncio.Queue[Tuple[str, str, Optional[str]]]] = None
_started = False
_workers: list[asyncio.Task] = []


def _conc() -> int:
    try:
        v = int(os.getenv("JINX_EMBED_REFRESH_CONC", "1") or "1")
        return max(1, min(4, v))
    except Exception:
        return 1


def _job_timeout_ms() -> int:
    try:
        return int(os.getenv("JINX_EMBED_REFRESH_JOB_MS", "900") or 900)
    except Exception:
        return 900


async def _worker() -> None:
    assert _Q is not None
    while True:
        abs_p, rel_p, sha = await _Q.get()
        try:
            # Admission guard for graph ops
            if _guard is not None:
                async with _guard("graph", timeout_ms=150) as admitted:
                    if not admitted:
                        continue
                    await _run_job(abs_p, rel_p, sha)
            else:
                await _run_job(abs_p, rel_p, sha)
        except Exception:
            pass
        finally:
            try:
                _Q.task_done()
            except Exception:
                pass


async def _run_job(abs_p: str, rel_p: str, sha: Optional[str]) -> None:
    if _embed_file is None:
        return
    # Compute sha if needed off-thread (best-effort)
    if not sha and _sha_path is not None:
        try:
            sha = await asyncio.to_thread(_sha_path, abs_p)
        except Exception:
            sha = None
    # Timebox the embed call
    try:
        timeout_s = max(0.1, _job_timeout_ms() / 1000.0)
        await asyncio.wait_for(_embed_file(abs_p, rel_p, file_sha=str(sha or "")), timeout=timeout_s)
    except Exception:
        # best-effort
        pass


async def _ensure_started() -> None:
    global _Q, _started, _workers
    if _started:
        return
    if _Q is None:
        _Q = asyncio.Queue(maxsize=64)
    # Launch N workers
    n = _conc()
    for _ in range(n):
        _workers.append(asyncio.create_task(_worker()))
    _started = True


async def enqueue_refresh(abs_path: str, rel_path: str, file_sha: Optional[str] = None) -> None:
    """Schedule an embedding refresh for file.

    - abs_path: absolute file path
    - rel_path: project-relative path
    - file_sha: optional sha256 of the file contents
    """
    await _ensure_started()
    assert _Q is not None
    try:
        await _Q.put((abs_path, rel_path, file_sha))
    except Exception:
        # queue full or cancelled; drop silently
        pass
