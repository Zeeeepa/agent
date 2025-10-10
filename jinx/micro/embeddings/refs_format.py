from __future__ import annotations

from typing import Optional, Tuple


def _code_block(snippet: str, lang: Optional[str]) -> str:
    body = snippet or ""
    return f"```{lang}\n{body}\n```" if lang else f"```\n{body}\n```"


def format_usage_ref(
    symbol: str,
    kind: Optional[str],
    file: str,
    line_start: int,
    line_end: int,
    snippet: str,
    lang: Optional[str],
    *,
    origin_file: str,
    origin_ls: int,
    origin_le: int,
) -> Tuple[str, str]:
    """Format a usage reference with explicit relation to the origin snippet.

    Header example:
    [usage|symbol: foo (function) | origin: pkg/mod.py:10-38 -> here: app/main.py:120-131]
    """
    sym = (symbol or "?").strip()
    k = (kind or "").strip()
    k_part = f" ({k})" if k else ""
    hdr = (
        f"[usage|symbol: {sym}{k_part} | origin: {origin_file}:{int(origin_ls)}-{int(origin_le)}"
        f" -> here: {file}:{int(line_start)}-{int(line_end)}]"
    )
    return (hdr, _code_block(snippet, lang))


essential_chars = 80

def format_literal_ref(
    query: str,
    file: str,
    line_start: int,
    line_end: int,
    preview: str,
    lang: Optional[str],
    *,
    origin_file: str,
    origin_ls: int,
    origin_le: int,
) -> Tuple[str, str]:
    """Format a literal-occurrence reference with connection back to origin snippet.

    Header example:
    [literal|q: "load_model" | origin: core/loader.py:50-90 -> here: tests/test_loader.py:22-25]
    """
    q = (query or "").strip().replace("\n", " ")
    if len(q) > essential_chars:
        q = q[: essential_chars - 1] + "…"
    hdr = (
        f"[literal|q: \"{q}\" | origin: {origin_file}:{int(origin_ls)}-{int(origin_le)}"
        f" -> here: {file}:{int(line_start)}-{int(line_end)}]"
    )
    return (hdr, _code_block(preview or "", lang))


__all__ = ["format_usage_ref", "format_literal_ref"]
