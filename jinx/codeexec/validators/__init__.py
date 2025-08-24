from __future__ import annotations

from typing import List, Callable, Optional
from .try_except import check_try_except
from .triple_quotes import check_triple_quotes

Checker = Callable[[str], Optional[str]]

_CHECKS: list[Checker] = [
    check_try_except,
    check_triple_quotes,
]


def collect_violations(code: str) -> List[str]:
    """Run all validators and return a list of violation messages."""
    msgs: List[str] = []
    for chk in _CHECKS:
        try:
            m = chk(code)
            if m:
                msgs.append(m)
        except Exception:
            # Validator failures are ignored to keep best-effort semantics
            pass
    return msgs
