from __future__ import annotations

from typing import Optional

from .config import is_enabled


def check_side_effect_policy(code: str) -> Optional[str]:
    """Enforce verification for OS side effects via sentinel prints.

    If the code attempts OS-level side effects (open browser/app/file) using
    common APIs but does not emit sentinel prints (OK:/ERROR: or <<JINX_ERROR>>),
    we flag a violation so the recovery flow can correct it.
    """
    if not is_enabled("side_effects", True):
        return None
    c = (code or "")
    low = c.lower()
    # Heuristics for side-effect intents
    uses_web = "webbrowser.open(" in c
    uses_startfile = "os.startfile(" in c
    # subprocess.Popen with xdg-open/open/cmd start
    uses_popen = "subprocess.Popen(" in c and ("xdg-open" in low or " open" in low or "cmd" in low and "/c" in low and "start" in low)
    side_effect = uses_web or uses_startfile or uses_popen
    if not side_effect:
        return None
    # Must contain at least one sentinel print
    has_ok = "OK:" in c
    has_err = "ERROR:" in c or "<<JINX_ERROR>>" in c
    if not (has_ok or has_err):
        return "side-effect must verify with sentinel prints (OK:/ERROR:)"
    return None
