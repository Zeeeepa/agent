"""Banner facade.

Thin wrapper delegating to micro-module implementation under
``jinx.micro.ui.banner``.
"""

from __future__ import annotations

from jinx.micro.ui.banner import show_banner as show_banner


__all__ = [
    "show_banner",
]
