from __future__ import annotations

import re
from typing import List, Tuple


def parse_tagged_blocks(out: str, code_id: str) -> List[Tuple[str, str]]:
    """Extract pairs of (tag, content) for the given code id.

    Tolerates CRLF and surrounding whitespace and captures minimal content.
    """
    pattern = rf"<(\w+)_{re.escape(code_id)}\s*>[\s\r\n]*" \
              rf"(.*?)" \
              rf"[\s\r\n]*</\1_{re.escape(code_id)}\s*>"
    return re.findall(pattern, out, re.S)
