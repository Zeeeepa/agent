from __future__ import annotations

import os
import re
from typing import Iterable, Pattern, List

# Defaults: STRICT privacy
# - internal filtering ON
# - absolute path redaction ON
# - PII redaction ON


def _is_on(env: str | None, default: bool = True) -> bool:
    val = (env if env is not None else ("1" if default else "0")).strip().lower()
    return val not in ("", "0", "false", "off", "no")


def filter_internals_enabled() -> bool:
    return _is_on(os.getenv("JINX_FILTER_INTERNALS"), True)


def filter_mode() -> str:
    m = (os.getenv("JINX_FILTER_MODE", "strip") or "strip").strip().lower()
    return "redact" if m == "redact" else "strip"


def restrict_abs_paths_enabled() -> bool:
    return _is_on(os.getenv("JINX_PRIVACY_ALLOW_ABS_PATHS"), False) is False


def pii_redact_enabled() -> bool:
    return _is_on(os.getenv("JINX_PRIVACY_PII_REDACT"), True)


# Common sensitive token patterns (best-effort, not exhaustive)
_PII_PATTERNS: List[Pattern[str]] = [
    # OpenAI keys
    re.compile(r"sk-[a-zA-Z0-9]{20,100}", re.IGNORECASE),
    # GitHub tokens
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,100}", re.IGNORECASE),
    # AWS Access Key IDs
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # Slack tokens
    re.compile(r"xox[abpr]-[A-Za-z0-9\-]{10,100}", re.IGNORECASE),
    # JWT-like long tokens
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
]


def pii_patterns() -> Iterable[Pattern[str]]:
    return _PII_PATTERNS


def redact_pii(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _PII_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


# Absolute path detectors (best-effort)
_WIN_ABS = re.compile(r"(?i)\b[A-Z]:\\")
_POSIX_ABS = re.compile(r"(^|\s)/(?:[^ \t\r\n]{1,256})")


def has_abs_path(s: str) -> bool:
    if not s:
        return False
    return bool(_WIN_ABS.search(s) or _POSIX_ABS.search(s))
