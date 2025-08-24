from __future__ import annotations

import os
from datetime import datetime
from jinx.log_paths import SANDBOX_DIR
import asyncio


def make_run_log_path(base_dir: str = SANDBOX_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"pending_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log")

async def async_rename_run_log(log_path: str, status: str) -> str:
    try:
        base = os.path.basename(log_path)
        ts = base.split("_", 1)[1][:-4] if ("_" in base and base.endswith(".log")) else datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        prefix = "ok" if status == "ok" else "error"
        new_path = os.path.join(os.path.dirname(log_path), f"{prefix}_{ts}.log")
        if os.path.abspath(new_path) == os.path.abspath(log_path):
            return log_path
        await asyncio.to_thread(os.replace, log_path, new_path)
        return new_path
    except Exception:
        return log_path


def read_latest_sandbox_tail(max_lines: int = 80) -> tuple[str | None, bool]:
    """Return the latest sandbox log content and whether it's a tail.

    Returns (content, tailed). If no logs found or on error, returns (None, False).
    """
    try:
        if not os.path.isdir(SANDBOX_DIR):
            return None, False
        logs = [
            os.path.join(SANDBOX_DIR, f)
            for f in os.listdir(SANDBOX_DIR)
            if f.endswith(".log")
        ]
        if not logs:
            return None, False
        latest = max(logs, key=os.path.getmtime)
        with open(latest, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        if len(lines) <= max_lines:
            return ("\n".join(lines), False)
        return ("\n".join(lines[-max_lines:]), True)
    except Exception:
        return None, False
