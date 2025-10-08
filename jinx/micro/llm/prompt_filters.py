from __future__ import annotations

import os
import re
from typing import Iterable

from jinx.micro.common.internal_paths import (
    is_internal_path,
    is_log_path,
    is_restricted_path,
)
from jinx.micro.privacy.policy import (
    filter_internals_enabled,
    filter_mode as privacy_filter_mode,
    restrict_abs_paths_enabled,
    has_abs_path,
    pii_redact_enabled,
    redact_pii,
)


def sanitize_prompt_for_external_api(text: str) -> str:
    """Strip any sections that reveal internal .jinx paths or artifacts.

    Strategy:
    - Remove snippet headers like "[path:ls-le]" when path includes .jinx and drop the following code block (```...```).
    - Remove any standalone lines that include .jinx absolute/relative paths.
    - Keep memory content (evergreen/compact) as-is, but without leaking file paths.
    - Controlled by env JINX_FILTER_INTERNALS (on by default).
    """
    # Internal path filtering toggle (STRICT by default)
    try:
        on = filter_internals_enabled()
    except Exception:
        on = True
    lines = (text or "").splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    mode = privacy_filter_mode()
    # Helper for redaction
    def _redact_line(ln: str) -> str:
        try:
            # replace path segments only; keep surrounding text
            ln2 = re.sub(r"(^|[\\/])\.jinx([\\/]|$)", r"\1[JINX]\2", ln, flags=re.IGNORECASE)
            ln2 = re.sub(r"(^|[\\/])log([\\/]|$)", r"\1[LOG]\2", ln2, flags=re.IGNORECASE)
            return ln2
        except Exception:
            return ln
    while i < n:
        ln = lines[i]
        # Matches snippet header like: "[path:ls-le]" when path includes .jinx and drop the following code block (```...```).
        m = re.match(r"^\[(?P<path>[^\]:]+):\d+-\d+\]\s*$", ln.strip())
        if m:
            p = m.group("path") or ""
            if is_restricted_path(p):
                # Skip this header and the following fenced code block if present
                i += 1
                if i < n and lines[i].lstrip().startswith("```"):
                    fence = lines[i].lstrip()[:3]
                    i += 1
                    # skip until closing fence
                    while i < n:
                        if lines[i].lstrip().startswith(fence):
                            i += 1
                            break
                        i += 1
                continue
        # Drop or redact any lines that directly reveal restricted paths (.jinx/log)
        if on and is_restricted_path(ln):
            if mode == "redact":
                out.append(_redact_line(ln))
            else:
                # strip mode (default)
                i += 1
                continue
        else:
            # Optionally redact/drop absolute OS paths (e.g., C:\..., /var/...)
            if restrict_abs_paths_enabled() and has_abs_path(ln):
                if mode == "redact":
                    # Coarse: mask drive/root indicators
                    ln_mask = re.sub(r"(?i)\b([A-Z]):\\", r"[DRIVE]\\", ln)
                    ln_mask = re.sub(r"(^|\s)/", r"\1[ROOT]/", ln_mask)
                    out.append(ln_mask)
                else:
                    i += 1
                    continue
            else:
                out.append(ln)
        i += 1
    sanitized = "\n".join(out)
    # PII redaction as a final pass
    if pii_redact_enabled():
        try:
            sanitized = redact_pii(sanitized)
        except Exception:
            pass
    return sanitized
