from __future__ import annotations

from typing import Iterable, List, Sequence

from .constants import chaos_taboo
from .errors import UnsafeCodeError


def find_violations(
    snippet: str,
    taboo: Sequence[str] | None = None,
    *,
    case_sensitive: bool = True,
) -> List[str]:
    """Return taboo substrings present in ``snippet``.

    Parameters
    ----------
    snippet : str
        The code snippet to check.
    taboo : Sequence[str] | None, optional
        The taboo list to use. Defaults to the module-level ``chaos_taboo``.
    case_sensitive : bool, optional
        Whether checks are case-sensitive. Default is True.

    Returns
    -------
    List[str]
        A list of taboo substrings found in the snippet (may be empty).
    """
    haystack = snippet if case_sensitive else snippet.lower()
    needles: Iterable[str] = (taboo or chaos_taboo)
    if not case_sensitive:
        needles = (n.lower() for n in needles)
    return [n for n in needles if n in haystack]


def is_code_safe(
    snippet: str,
    taboo: Sequence[str] | None = None,
    *,
    case_sensitive: bool = True,
) -> bool:
    """Return ``True`` iff ``snippet`` contains no taboo substrings.

    This is a convenience wrapper around ``find_violations``.
    """
    return len(find_violations(snippet, taboo, case_sensitive=case_sensitive)) == 0


def assert_code_safe(
    snippet: str,
    taboo: Sequence[str] | None = None,
    *,
    case_sensitive: bool = True,
) -> None:
    """Raise ``UnsafeCodeError`` if ``snippet`` violates taboo policy.

    Parameters
    ----------
    snippet : str
        Code to check.
    taboo : Sequence[str] | None, optional
        Custom taboo list. Defaults to the module-level ``chaos_taboo``.
    case_sensitive : bool, optional
        Whether to treat checks as case-sensitive.

    Raises
    ------
    UnsafeCodeError
        If any taboo substring is detected.
    """
    violations = find_violations(snippet, taboo, case_sensitive=case_sensitive)
    if violations:
        joined = ", ".join(sorted(set(violations)))
        raise UnsafeCodeError(
            f"Unsafe code detected; taboo substrings present: {joined}",
            violations=violations,
            snippet=snippet,
        )
