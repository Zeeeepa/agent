from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

__all__ = [
    "ResultBase",
    "PatchResult",
    "VerifyResult",
    "RefactorResult",
]


@dataclass
class ResultBase:
    ok: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_warning(self, w: str) -> None:
        if w:
            self.warnings.append(str(w))

    def add_error(self, e: str) -> None:
        if e:
            self.errors.append(str(e))

    @property
    def reason(self) -> str:
        if self.errors:
            return self.errors[0]
        return ""


@dataclass
class PatchResult(ResultBase):
    paths: List[str] = field(default_factory=list)
    diff: str = ""
    strategy: str = ""


@dataclass
class VerifyResult(ResultBase):
    goal: str = ""
    files: List[str] = field(default_factory=list)
    score: Optional[float] = None


@dataclass
class RefactorResult(ResultBase):
    ops_count: int = 0
    changed_files: List[str] = field(default_factory=list)
