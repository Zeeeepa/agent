from __future__ import annotations

import re

from jinx.config import ALL_TAGS


def sanitize_transcript_for_memory(mem_text: str, last_user_line: str) -> str:
    """Sanitize transcript text for inclusion in <memory>.

    - Drop the latest user line if it was just entered (avoid duplication).
    - Remove tool blocks like <machine_...>...</machine_...>, <python_...>...</python_...>.
    - Remove header blocks (<embeddings_context>, <memory>, <evergreen>, <task>, <error>).
    - Collapse excessive blank lines.
    """
    t = mem_text or ""
    try:
        if last_user_line:
            lines = [ln for ln in t.splitlines()]
            if lines:
                last = lines[-1].strip()
                # consider both labeled and unlabeled forms
                if (
                    last == last_user_line
                    or last == f"User: {last_user_line}".strip()
                    or (last.lower().startswith("user:") and last[5:].strip() == last_user_line)
                ):
                    lines = lines[:-1]
                    t = "\n".join(lines)
    except Exception:
        pass
    try:
        tag_alt = "|".join(sorted(ALL_TAGS))
        tool_pat = re.compile(fr"<(?:{tag_alt})_[^>]+>.*?</(?:{tag_alt})_[^>]+>", re.DOTALL)
        t = tool_pat.sub("", t)
        header_pat = re.compile(r"<(?:embeddings_context|memory|evergreen|task|error)>.*?</(?:embeddings_context|memory|evergreen|task|error)>", re.DOTALL)
        t = header_pat.sub("", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
    except Exception:
        pass
    return t
