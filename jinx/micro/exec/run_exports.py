from __future__ import annotations

import os
import json
import time
from typing import Optional, Tuple, List

from jinx.micro.memory.storage import memory_dir


def _exports_dir() -> str:
    try:
        base = os.getenv("JINX_EXPORTS_DIR")
        if base:
            return base
    except Exception:
        pass
    return os.path.join(memory_dir(), "exports")


def _paths() -> Tuple[str, str, str, str]:
    root = _exports_dir()
    os.makedirs(root, exist_ok=True)
    return (
        os.path.join(root, "last_run_stdout.txt"),
        os.path.join(root, "last_run_stderr.txt"),
        os.path.join(root, "last_run_status.json"),
        os.path.join(root, "last_run_log.txt"),
    )


def _write_atomic(path: str, data: str) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        pass


def write_last_run(stdout_text: str | None, stderr_text: str | None, log_path: str | None, ok: bool) -> None:
    """Persist last run artifacts for prompt macros to consume (best-effort)."""
    out_p, err_p, st_p, log_p = _paths()
    now = int(time.time() * 1000)
    try:
        max_bytes = int(os.getenv("JINX_RUN_EXPORT_MAX_BYTES", "65536"))
    except Exception:
        max_bytes = 65536

    def _clamp(s: Optional[str]) -> str:
        t = (s or "").strip()
        if max_bytes > 0 and len(t.encode("utf-8", errors="ignore")) > max_bytes:
            # naive byte clamp by slicing chars; good enough for logs
            t = t[-max_bytes:]
        return t

    _write_atomic(out_p, _clamp(stdout_text))
    _write_atomic(err_p, _clamp(stderr_text))
    # persist last known log path to a simple text for easy access
    _write_atomic(log_p, (log_path or "").strip())
    st = {
        "ts": now,
        "status": "ok" if ok else "error",
        "log_path": (log_path or "").strip(),
    }
    try:
        _write_atomic(st_p, json.dumps(st, ensure_ascii=False))
    except Exception:
        pass


def _is_fresh(ts_ms: int, ttl_ms: int) -> bool:
    if ttl_ms <= 0:
        return True
    now = int(time.time() * 1000)
    return (now - ts_ms) <= ttl_ms


def _read_status() -> Tuple[int, str, str]:
    _out, _err, st_p, log_p = _paths()
    try:
        with open(st_p, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return int(obj.get("ts") or 0), str(obj.get("status") or ""), str(obj.get("log_path") or "")
    except Exception:
        pass
    # fallback to mtime of stdout if status missing
    out_p, _e, _s, _l = _paths()
    try:
        ts = int(os.stat(out_p).st_mtime * 1000)
    except Exception:
        ts = 0
    return ts, "", ""


def read_last_stdout(n_lines: int, preview_chars: int, ttl_ms: int) -> str:
    out_p, _e, _s, _l = _paths()
    ts, status, _log = _read_status()
    if not _is_fresh(ts, ttl_ms):
        return ""
    try:
        with open(out_p, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = lines[-max(1, n_lines):]
    joined = " ".join(tail)
    return ("Run stdout: " + joined[:max(24, preview_chars)])


def read_last_stderr(n_lines: int, preview_chars: int, ttl_ms: int) -> str:
    _o, err_p, _s, _l = _paths()
    ts, status, _log = _read_status()
    if not _is_fresh(ts, ttl_ms):
        return ""
    try:
        with open(err_p, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = lines[-max(1, n_lines):]
    joined = " ".join(tail)
    return ("Run stderr: " + joined[:max(24, preview_chars)])


def read_last_status(ttl_ms: int) -> str:
    ts, status, log_path = _read_status()
    if not _is_fresh(ts, ttl_ms):
        return ""
    if not status:
        return ""
    return f"Run status: {status}"
