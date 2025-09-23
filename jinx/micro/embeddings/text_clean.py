from __future__ import annotations

import re
from jinx.config import ALL_TAGS

_RE_CODEY = re.compile(r"^\s*(print\s*\(|return\b|def\b|class\b|import\b|from\b)", re.I)
_RE_NUMBERY = re.compile(r"^[\s\d+\-*/().]+$")


def is_noise_text(pv: str) -> bool:
    pv = (pv or "").strip()
    if len(pv) < 4:
        return True
    if _RE_CODEY.match(pv):
        return True
    if _RE_NUMBERY.match(pv) and any(ch.isdigit() for ch in pv):
        return True
    return False


# Remove known wrapper tags like <machine_123>, </machine_123>, <python_...>
_TAG_OPEN_RE = re.compile(r"<([a-zA-Z_]+)(?:_\d+)?\s*>")
_TAG_CLOSE_RE = re.compile(r"</([a-zA-Z_]+)(?:_\d+)?\s*>")


def strip_known_tags(text: str) -> str:
    if not text:
        return text

    def repl_open(m: re.Match) -> str:
        base = m.group(1).lower()
        return "" if base in ALL_TAGS else m.group(0)

    def repl_close(m: re.Match) -> str:
        base = m.group(1).lower()
        return "" if base in ALL_TAGS else m.group(0)

    cleaned = _TAG_OPEN_RE.sub(repl_open, text)
    cleaned = _TAG_CLOSE_RE.sub(repl_close, cleaned)
    # Also drop lines that are only leftover angle brackets or whitespace
    cleaned_lines = []
    for ln in cleaned.splitlines():
        s = ln.strip()
        if not s:
            continue
        # drop a line if it became just <> or similar
        if s in {"<>", "</>", "<>:"}:
            continue
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines)
