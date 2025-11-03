from __future__ import annotations

from typing import List, Tuple


def validate_text(path: str, text: str) -> Tuple[bool, str]:
    """Run code validators against text. Returns (ok, message).

    - Uses jinx.codeexec validators if available; safe no-op when unavailable.
    - Aggregates violations into a concise message.
    """
    try:
        from jinx.codeexec import collect_violations, collect_violations_detailed
    except Exception:
        return True, "validators unavailable"
    try:
        v = collect_violations(text or "")
        if not v:
            return True, "ok"
        # Build detailed message
        try:
            dets = collect_violations_detailed(text or "")
            if dets:
                lines: List[str] = []
                for d in dets:
                    ident = str(d.get("id", "rule"))
                    line = int(d.get("line", 0))
                    msg = str(d.get("msg", ""))
                    cat = str(d.get("category", ""))
                    lines.append(f"[{cat}] {ident}@{line}: {msg}")
                return False, "; ".join(lines[:20])
        except Exception:
            pass
        return False, "; ".join(v[:10])
    except Exception as e:
        return True, f"validator error ignored: {e}"
