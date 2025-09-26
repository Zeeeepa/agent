from __future__ import annotations

import os


def _is_on(val: str | None) -> bool:
    return (val or "0").strip().lower() in {"1", "true", "yes", "on"}


def _auto_max_concurrency() -> int:
    cpus = os.cpu_count() or 2
    # Scale modestly with CPU to reduce IO/memory pressure
    return max(1, min(8, cpus // 2))


# Core toggles and parameters
# Default to enabled if the env var is absent, so embeddings are always on by default
ENABLE = _is_on(os.getenv("EMBED_PROJECT_ENABLE", "1"))
ROOT = os.getenv("EMBED_PROJECT_ROOT", os.getcwd())
SCAN_INTERVAL_MS = int(os.getenv("EMBED_PROJECT_SCAN_INTERVAL_MS", "2500"))
MAX_CONCURRENCY = int(os.getenv("EMBED_PROJECT_MAX_CONCURRENCY", str(_auto_max_concurrency())))
USE_WATCHDOG = _is_on(os.getenv("EMBED_PROJECT_USE_WATCHDOG", "1"))
MAX_FILE_BYTES = int(os.getenv("EMBED_PROJECT_MAX_FILE_BYTES", str(1_500_000)))
RECONCILE_SEC = int(os.getenv("EMBED_PROJECT_RECONCILE_SEC", "60"))

# Include/exclude
_INCLUDE_EXTS = os.getenv(
    "EMBED_PROJECT_INCLUDE_EXTS",
    "py,md,txt,js,ts,tsx,json,yaml,yml,ini,toml,sh,bat,ps1,go,rs,java,cs,cpp,c,h,jsx,tsx,sql,proto,gradle,kts,rb,php",
).strip()
INCLUDE_EXTS: list[str] = [x.strip().lower() for x in _INCLUDE_EXTS.split(",") if x.strip()]

_EXCLUDE_DIRS = os.getenv(
    "EMBED_PROJECT_EXCLUDE_DIRS",
    ".git,.hg,.svn,.venv,venv,node_modules,emb,log,__pycache__,dist,build,.idea,.vscode,.pytest_cache,.mypy_cache,.ruff_cache,__pypackages__",
).strip()
EXCLUDE_DIRS: list[str] = [x.strip() for x in _EXCLUDE_DIRS.split(",") if x.strip()]


__all__ = [
    "ENABLE",
    "ROOT",
    "SCAN_INTERVAL_MS",
    "MAX_CONCURRENCY",
    "USE_WATCHDOG",
    "MAX_FILE_BYTES",
    "RECONCILE_SEC",
    "INCLUDE_EXTS",
    "EXCLUDE_DIRS",
]
