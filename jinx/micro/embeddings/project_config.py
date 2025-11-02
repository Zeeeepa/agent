from __future__ import annotations

import os
from typing import Optional


def _is_on(val: str | None) -> bool:
    return (val or "0").strip().lower() in {"1", "true", "yes", "on"}


def _auto_max_concurrency() -> int:
    cpus = os.cpu_count() or 2
    # Scale modestly with CPU to reduce IO/memory pressure
    return max(1, min(8, cpus // 2))


# -----------------------------
# Project root resolution (RT)
# -----------------------------

_MARKERS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "Pipfile",
    "poetry.lock",
    "setup.py",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "gradlew",
    "Makefile",
    ".idea",
    ".vscode",
)


def _has_marker(dir_path: str) -> bool:
    try:
        for m in _MARKERS:
            p = os.path.join(dir_path, m)
            if os.path.isdir(p) or os.path.isfile(p):
                return True
        return False
    except Exception:
        return False


def _ascend_to_marker_root(start: str) -> Optional[str]:
    try:
        cur = os.path.abspath(start or os.getcwd())
        best = None
        while True:
            if _has_marker(cur):
                best = cur
            parent = os.path.abspath(os.path.join(cur, os.pardir))
            if parent == cur:
                break
            cur = parent
        return best
    except Exception:
        return None


def _normalize_root(p: str) -> str:
    try:
        return os.path.abspath(os.path.expanduser(p))
    except Exception:
        return p


def resolve_project_root() -> str:
    """Resolve the project root directory robustly.

    Priority:
    1. If EMBED_PROJECT_ROOT_MODE=env_only and EMBED_PROJECT_ROOT is set -> use it.
    2. If EMBED_PROJECT_ROOT is set -> use it.
    3. Else try to ascend from CWD to a directory containing known markers (VCS/build files).
    4. Fallback to CWD.
    """
    mode = (os.getenv("EMBED_PROJECT_ROOT_MODE", "auto") or "auto").strip().lower()
    env_root = os.getenv("EMBED_PROJECT_ROOT")
    if mode == "env_only":
        return _normalize_root(env_root or os.getcwd())
    if env_root:
        return _normalize_root(env_root)
    # auto: detect by markers from cwd
    best = _ascend_to_marker_root(os.getcwd())
    if best:
        return _normalize_root(best)
    return _normalize_root(os.getcwd())


# Core toggles and parameters
# Default to enabled if the env var is absent, so embeddings are always on by default
ENABLE = _is_on(os.getenv("EMBED_PROJECT_ENABLE", "1"))
ROOT = resolve_project_root()
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
    ".git,.hg,.svn,.venv,venv,node_modules,emb,log,.jinx,__pycache__,dist,build,.idea,.vscode,.pytest_cache,.mypy_cache,.ruff_cache,__pypackages__",
).strip()
EXCLUDE_DIRS: list[str] = [x.strip() for x in _EXCLUDE_DIRS.split(",") if x.strip()]
# Always exclude internal directories even if env overrides
for _dir in (".jinx", "log"):
    if _dir not in EXCLUDE_DIRS:
        EXCLUDE_DIRS.append(_dir)


__all__ = [
    "ENABLE",
    "ROOT",
    "resolve_project_root",
    "SCAN_INTERVAL_MS",
    "MAX_CONCURRENCY",
    "USE_WATCHDOG",
    "MAX_FILE_BYTES",
    "RECONCILE_SEC",
    "INCLUDE_EXTS",
    "EXCLUDE_DIRS",
]
