from __future__ import annotations

import re
from typing import List, Tuple

from jinx.micro.parser.api import parse_tagged_blocks as _parse_blocks


def _wrap_code_fence(tag: str, core: str) -> str:
    body = (core or "").strip("\n")
    # Ensure a python fence and preserve body as-is
    return f"<{tag}>\n```python\n{body}\n```\n</{tag}>"


def normalize_output_blocks(out: str, code_id: str) -> str:
    """Normalize model output to contain exactly one <python_{code_id}> block.

    - If multiple python blocks exist: pick the best by simple heuristics and drop others.
    - If the chosen block lacks fences: wrap it in ```python fences.
    - If no python block: return original output.
    - Preserve order of non-code blocks.
    """
    try:
        pairs: List[Tuple[str, str]] = _parse_blocks(out or "", code_id)
    except Exception:
        return out or ""
    if not pairs:
        return out or ""

    py_tag = f"python_{code_id}"
    py_blocks: List[Tuple[int, str]] = []  # (index, core)
    for i, (tag, core) in enumerate(pairs):
        if tag.strip() == py_tag:
            py_blocks.append((i, core))

    if not py_blocks:
        return out or ""

    # Heuristics to pick best block: prefer ones containing defs/imports/classes, else longest
    def _score_code(s: str) -> Tuple[int, int]:
        txt = s or ""
        sig = 0
        for kw in ("def ", "class ", "import ", "from "):
            if kw in txt:
                sig += 1
        return (sig, len(txt))

    best_idx, best_core = max(py_blocks, key=lambda ic: _score_code(ic[1]))

    # Rebuild output: keep order, but replace the selected python block with fenced version,
    # and drop other python blocks with same code_id.
    out_parts: List[str] = []
    for i, (tag, core) in enumerate(pairs):
        if tag.strip() == py_tag:
            if i == best_idx:
                # Ensure fenced
                has_fence = bool(re.search(r"```", core or ""))
                if has_fence:
                    out_parts.append(f"<{tag}>\n{core}\n</{tag}>")
                else:
                    out_parts.append(_wrap_code_fence(tag, core))
            # Skip other python_{code_id} occurrences
            continue
        # Keep other blocks as-is
        out_parts.append(f"<{tag}>\n{(core or '').strip()}\n</{tag}>")

    return "\n".join(out_parts)
