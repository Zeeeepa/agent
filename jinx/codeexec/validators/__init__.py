from __future__ import annotations

from typing import List, Callable, Optional
from .try_except_ast import check_try_except_ast as check_try_except
from .triple_quotes import check_triple_quotes
from .nontrivial import check_nontrivial
from .side_effects import check_side_effect_policy
from .comment_only import check_comment_only
from .banned_dyn import check_banned_dynamic
from .io_clamps import check_io_clamps
from .net_safety import check_net_system_safety
from .fs_safety import check_fs_safety
from .rt_limits import check_rt_limits
from .blocking_io import check_blocking_io
from .http_safety import check_http_safety
from .spawn_policy import check_spawn_policy
from .import_policy import check_import_policy
from .import_star import check_import_star
from .deserialization_safety import check_deserialization_safety

Checker = Callable[[str], Optional[str]]

_CHECKS: list[Checker] = [
    check_try_except,
    check_triple_quotes,
    check_nontrivial,
    check_side_effect_policy,
    check_comment_only,
    check_banned_dynamic,
    check_deserialization_safety,
    check_io_clamps,
    check_net_system_safety,
    check_http_safety,
    check_import_policy,
    check_import_star,
    check_spawn_policy,
    check_fs_safety,
    check_rt_limits,
    check_blocking_io,
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
