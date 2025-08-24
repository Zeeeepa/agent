from __future__ import annotations

from typing import Sequence


class UnsafeCodeError(RuntimeError):
    """Exception raised when a code snippet violates taboo checks.

    Attributes
    ----------
    violations : Sequence[str]
        The taboo substrings that were found in the snippet.
    snippet_preview : str
        A trimmed preview of the offending code to assist debugging.
    """

    def __init__(self, message: str, *, violations: Sequence[str], snippet: str) -> None:
        super().__init__(message)
        self.violations: Sequence[str] = violations
        # Keep previews compact to avoid log spam and leaking large payloads
        self.snippet_preview: str = snippet[:500]
