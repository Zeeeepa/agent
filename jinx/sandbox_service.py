"""Sandbox service facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.sandbox.service`` to keep the public API stable.
"""

from __future__ import annotations

from jinx.micro.sandbox.service import (
    blast_zone as blast_zone,
    arcane_sandbox as arcane_sandbox,
)


__all__ = [
    "blast_zone",
    "arcane_sandbox",
]
