from __future__ import annotations

from typing import Optional
import io
import tokenize
from .config import is_enabled


def check_comment_only(code: str) -> Optional[str]:
    """Flag blocks that contain only comments/blank lines.

    Uses tokenization and ignores ENCODING, NL/NEWLINE, INDENT/DEDENT, and COMMENT
    tokens. If nothing meaningful remains, it's trivial and should trigger recovery.
    """
    if not is_enabled("comment_only", True):
        return None
    src = code or ""
    try:
        g = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok in g:
            if tok.type in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                continue
            # Any other token means there is real code
            return None
        return "comment-only code block"
    except Exception:
        # On tokenizer failure, do not flag here (other validators handle syntax)
        return None
