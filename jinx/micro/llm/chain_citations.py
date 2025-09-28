from __future__ import annotations

from typing import Any, Dict, List


def build_citation_block(evidence: Dict[str, Any], *, max_dialogue: int = 3, max_code: int = 3) -> str:
    """Render a compact citations block from gathered evidence.

    Returns a string like:
    <plan_citations>
    [d1] source_name (score=0.82)
    [d2] ...
    [c1] path/to/file.py (score=0.77)
    ...
    </plan_citations>
    """
    if not evidence:
        return ""
    queries = evidence.get("queries") or []
    d_items: List[tuple[float, str]] = []
    c_items: List[tuple[float, str]] = []
    for q in queries:
        for rec in (q.get("dialogue") or []):
            try:
                d_items.append((float(rec.get("score") or 0.0), str(rec.get("src") or "")))
            except Exception:
                continue
        for rec in (q.get("code") or []):
            try:
                c_items.append((float(rec.get("score") or 0.0), str(rec.get("src") or "")))
            except Exception:
                continue
    if not d_items and not c_items:
        return ""
    # Sort by score descending and dedupe by src
    def _dedupe_top(items: List[tuple[float, str]], k: int) -> List[tuple[float, str]]:
        seen: set[str] = set()
        out: List[tuple[float, str]] = []
        for s, src in sorted(items, key=lambda x: x[0], reverse=True):
            if not src or src in seen:
                continue
            seen.add(src)
            out.append((s, src))
            if len(out) >= k:
                break
        return out

    d_top = _dedupe_top(d_items, max_dialogue)
    c_top = _dedupe_top(c_items, max_code)
    lines: List[str] = []
    idx = 1
    for s, src in d_top:
        lines.append(f"[d{idx}] {src} (score={s:.2f})")
        idx += 1
    idx = 1
    for s, src in c_top:
        lines.append(f"[c{idx}] {src} (score={s:.2f})")
        idx += 1
    if not lines:
        return ""
    return "<plan_citations>\n" + "\n".join(lines) + "\n</plan_citations>"
