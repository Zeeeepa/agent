from __future__ import annotations

from typing import Optional

from jinx.micro.parser.api import parse_tagged_blocks


def attach_error_code(err_msg: str, out: Optional[str], code_id: str, *, code_body: Optional[str] = None) -> str:
    """Build an error payload that includes the executed code under <error_code>.

    - If code_body is provided, use it; otherwise extract <python_{code_id}> from 'out'.
    - Returns the payload string to pass to corrupt_report().
    """
    payload = (err_msg or "").strip()
    body = (code_body or "").strip()
    if not body and (out or "").strip():
        try:
            pairs = parse_tagged_blocks(out or "", code_id)
        except Exception:
            pairs = []
        tag_name = f"python_{code_id}"
        for tag, core in pairs:
            if tag == tag_name:
                body = (core or "").strip()
                break
    if body:
        payload = f"{payload}\n\n<error_code>\n{body}\n</error_code>"
    return payload


__all__ = ["attach_error_code"]
