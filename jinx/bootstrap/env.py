from __future__ import annotations

from typing import Iterable
from .deps import ensure_optional

# Try to ensure python-dotenv is available at runtime; if not, proceed with noop
try:
    _mods = ensure_optional(["dotenv"])  # installs if missing
    dotenv = _mods.get("dotenv")  # type: ignore[assignment]
except Exception:
    dotenv = None  # type: ignore[assignment]


def load_env(paths: Iterable[str] | None = None) -> None:
    """Best-effort load of environment variables via python-dotenv.

    If python-dotenv is unavailable, this is a no-op.
    """
    if dotenv is None:
        return
    if paths:
        for p in paths:
            try:
                dotenv.load_dotenv(p, override=False)
            except Exception:
                pass
        return
    try:
        found = dotenv.find_dotenv(usecwd=True)
        if found:
            dotenv.load_dotenv(found, override=False)
        else:
            dotenv.load_dotenv(override=False)
    except Exception:
        pass
    
    # Apply autonomous configuration after .env is loaded
    try:
        from jinx.micro.runtime.autoconfig import apply_auto_defaults
        apply_auto_defaults()
    except Exception:
        # Non-fatal: continue without auto-config
        pass
