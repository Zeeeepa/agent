from __future__ import annotations

"""Utilities for assembling and normalizing conversation headers.

Provides helpers to:
- normalize unicode/nbspace characters
- enforce blank-line separation between known header blocks
- build the standardized header text from optional ctx/memory/task
"""

import re
from typing import Optional, List

from .format_normalization import normalize_unicode_spaces


def ensure_header_block_separation(text: str) -> str:
    """Ensure blank lines between specific header blocks.

    Currently enforces:
    - </embeddings_context>  [blank line]  <evergreen>
    - </embeddings_context>  [blank line]  <embeddings_code>
    - </embeddings_context>  [blank line]  <embeddings_refs>
    - </embeddings_code>     [blank line]  <evergreen>
    - </embeddings_code>     [blank line]  <embeddings_refs>
    - </embeddings_refs>     [blank line]  <evergreen>
    - </embeddings_refs>     [blank line]  <memory>
    - </embeddings_refs>     [blank line]  <task>
    - </embeddings_refs>     [blank line]  <error>
    - </evergreen>           [blank line]  <memory>
    - </memory>              [blank line]  <task>
    - </task>                [blank line]  <error>
    """
    t = normalize_unicode_spaces(text)
    # Normalize any whitespace (including existing newlines) between blocks to exactly one blank line
    t = re.sub(r"(</embeddings_context>)[\s\u00A0\u2007\u202F]*(<evergreen>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_context>)[\s\u00A0\u2007\u202F]*(<embeddings_code>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_context>)[\s\u00A0\u2007\u202F]*(<embeddings_refs>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_context>)[\s\u00A0\u2007\u202F]*(<embeddings_memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_context>)[\s\u00A0\u2007\u202F]*(<plan_kernels>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<evergreen>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<embeddings_refs>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<embeddings_memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<plan_kernels>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_refs>)[\s\u00A0\u2007\u202F]*(<evergreen>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_refs>)[\s\u00A0\u2007\u202F]*(<plan_kernels>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_refs>)[\s\u00A0\u2007\u202F]*(<embeddings_memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<task>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<error>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_memory>)[\s\u00A0\u2007\u202F]*(<evergreen>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_memory>)[\s\u00A0\u2007\u202F]*(<embeddings_code>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_memory>)[\s\u00A0\u2007\u202F]*(<embeddings_refs>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_memory>)[\s\u00A0\u2007\u202F]*(<memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_memory>)[\s\u00A0\u2007\u202F]*(<task>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_memory>)[\s\u00A0\u2007\u202F]*(<error>)", r"\1\n\n\2", t)
    t = re.sub(r"(</evergreen>)[\s\u00A0\u2007\u202F]*(<memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</plan_kernels>)[\s\u00A0\u2007\u202F]*(<evergreen>)", r"\1\n\n\2", t)
    t = re.sub(r"(</plan_kernels>)[\s\u00A0\u2007\u202F]*(<memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</plan_kernels>)[\s\u00A0\u2007\u202F]*(<task>)", r"\1\n\n\2", t)
    t = re.sub(r"(</plan_kernels>)[\s\u00A0\u2007\u202F]*(<error>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_refs>)[\s\u00A0\u2007\u202F]*(<memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_refs>)[\s\u00A0\u2007\u202F]*(<task>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_refs>)[\s\u00A0\u2007\u202F]*(<error>)", r"\1\n\n\2", t)
    t = re.sub(r"(</memory>)[\s\u00A0\u2007\u202F]*(<task>)", r"\1\n\n\2", t)
    t = re.sub(r"(</task>)[\s\u00A0\u2007\u202F]*(<error>)", r"\1\n\n\2", t)
    return t


def build_header(ctx: str | None, mem_text: str | None, task_text: str | None, error_text: str | None = None, evergreen_text: str | None = None) -> str:
    """Build the standardized header text from parts.

    - "ctx" may include one or both blocks already wrapped: <embeddings_context>... and/or <embeddings_code>...
    - Each provided block is wrapped (or assumed wrapped) and joined with clean spacing.
    - Final output guarantees proper separation between blocks.
    """
    parts: List[str] = []

    def _is_wrapped(s: str, tag: str) -> bool:
        if not s:
            return False
        ss = s.strip()
        return ss.startswith(f"<{tag}") and ss.endswith(f"</{tag}>")

    def _wrap_if_needed(s: str, tag: str) -> str:
        core = (s or "").strip()
        if not core:
            return ""
        return core if _is_wrapped(core, tag) else f"<{tag}>\n{core}\n</{tag}>"
    if ctx:
        parts.append(ctx.rstrip() + "\n")  # ctx is already wrapped as <embeddings_context>...</embeddings_context>
    if evergreen_text and evergreen_text.strip():
        ev = evergreen_text.strip()
        ev_block = ev if _is_wrapped(ev, "evergreen") else f"<evergreen>\n{ev}\n</evergreen>"
        parts.append(ev_block + "\n")
    if mem_text and mem_text.strip():
        mt = mem_text.strip()
        mem_block = mt if _is_wrapped(mt, "memory") else f"<memory>\n{mt}\n</memory>"
        parts.append(mem_block + "\n")
    if task_text and task_text.strip():
        parts.append(f"<task>\n{task_text.strip()}\n</task>\n")
    if error_text and error_text.strip():
        parts.append(f"<error>\n{error_text.strip()}\n</error>\n")

    if not parts:
        return ""

    header_text = "\n".join(parts)

    # Collapse accidental nested identical blocks like <memory><memory>...</memory></memory>
    def _collapse_nested(text: str, tag: str) -> str:
        # Repeat until no more nested patterns are found
        pat = re.compile(rf"<{tag}[^>]*>\s*<\s*{tag}[^>]*>([\s\S]*?)</\s*{tag}\s*>\s*</\s*{tag}\s*>", re.IGNORECASE)
        prev = None
        cur = text
        while prev != cur:
            prev = cur
            cur = pat.sub(rf"<{tag}>\\1</{tag}>", cur)
        return cur

    header_text = _collapse_nested(header_text, "memory")
    header_text = _collapse_nested(header_text, "evergreen")

    header_text = ensure_header_block_separation(header_text)
    return header_text
