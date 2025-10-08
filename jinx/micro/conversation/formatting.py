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
    t = re.sub(r"(</embeddings_context>)[\s\u00A0\u2007\u202F]*(<plan_kernels>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<evergreen>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<embeddings_refs>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<plan_kernels>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_refs>)[\s\u00A0\u2007\u202F]*(<evergreen>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_refs>)[\s\u00A0\u2007\u202F]*(<plan_kernels>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<memory>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<task>)", r"\1\n\n\2", t)
    t = re.sub(r"(</embeddings_code>)[\s\u00A0\u2007\u202F]*(<error>)", r"\1\n\n\2", t)
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
    if ctx:
        parts.append(ctx.rstrip() + "\n")  # ctx is already wrapped as <embeddings_context>...</embeddings_context>
    if evergreen_text and evergreen_text.strip():
        parts.append(f"<evergreen>\n{evergreen_text.strip()}\n</evergreen>\n")
    if mem_text and mem_text.strip():
        parts.append(f"<memory>\n{mem_text.strip()}\n</memory>\n")
    if task_text and task_text.strip():
        parts.append(f"<task>\n{task_text.strip()}\n</task>\n")
    if error_text and error_text.strip():
        parts.append(f"<error>\n{error_text.strip()}\n</error>\n")

    if not parts:
        return ""

    header_text = "\n".join(parts)
    header_text = ensure_header_block_separation(header_text)
    return header_text
