from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass(slots=True)
class MemoryItem:
    """A normalized memory item for hierarchical levels.

    - text: displayable text line/paragraph (already trimmed as needed)
    - source: logical source, e.g. 'compact', 'evergreen', 'state', 'dialogue', 'kb', 'summary'
    - ts_ms: timestamp in ms (0 if unknown)
    - meta: arbitrary metadata (e.g., preview hash, channel, path)
    """
    text: str
    source: str
    ts_ms: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Triplet:
    """Structured knowledge triple.

    - subject, predicate, object
    - count: frequency observed (optional)
    - last_ts: last seen timestamp (ms)
    - meta: extra fields
    """
    subject: str
    predicate: str
    object: str
    count: int = 0
    last_ts: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        return f"kb: {self.subject} | {self.predicate} | {self.object}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
