from __future__ import annotations

from typing import Tuple


def extract(tag: str, text: str) -> str | None:
    a = text.find(f"<{tag}>")
    if a == -1:
        return None
    b = text.find(f"</{tag}>", a)
    if b == -1:
        return None
    return text[a + len(tag) + 2 : b].strip()


def parse_output(model_out: str) -> Tuple[str, str | None]:
    compact = extract("mem_compact", model_out)
    durable = extract("mem_evergreen", model_out)
    if compact is None:
        compact = model_out.strip()
    return compact, durable
