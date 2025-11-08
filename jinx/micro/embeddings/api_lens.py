from __future__ import annotations

import re
from typing import List, Tuple


_METHODS = ("get", "post", "put", "patch", "delete")


def find_endpoints(code: str) -> List[Tuple[str, str, str]]:
    """Extract (method, path, handler) from FastAPI/Router decorators.
    Best-effort regex; safe and fast.
    """
    endpoints: List[Tuple[str, str, str]] = []
    if not code:
        return endpoints
    # Patterns: @app.get("/path") or @router.post('/x') over def handler(...)
    for m in re.finditer(r"@\s*(?:app|router)\.(get|post|put|patch|delete)\(\s*([\"\'])(.+?)\2\s*\).*?\n\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\(", code, re.DOTALL):
        method = (m.group(1) or "").upper()
        path = m.group(3) or ""
        handler = m.group(4) or "handler"
        endpoints.append((method, path, handler))
    return endpoints


def find_models(code: str) -> List[str]:
    """Heuristic Pydantic model names appearing in type annotations or Body(..).
    Returns unique list of identifiers.
    """
    if not code:
        return []
    names = set()
    # Type annotation capture: ": ModelName" or "-> ModelName"
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]*)\b", code):
        tok = m.group(1)
        # Basic filter to reduce noise
        if tok and tok[0].isupper() and len(tok) >= 3:
            names.add(tok)
    # Prefer a small cap
    out = list(names)[:12]
    out.sort()
    return out


def extract_api_edges(file_rel: str, header: str, code_block: str) -> Tuple[str, str] | None:
    """Return (hdr, block) for API graph enrichment, or None.
    Header format: [api:<file_rel>]
    Block: bullet lines of endpoints and referenced models.
    """
    eps = find_endpoints(code_block)
    if not eps:
        return None
    models = find_models(code_block)
    lines: List[str] = []
    lines.append("Endpoints:")
    for (method, path, handler) in eps[:10]:
        lines.append(f"- {method} {path} -> {handler}()")
    if models:
        lines.append("Models:")
        for m in models[:12]:
            lines.append(f"- {m}")
    body = "\n".join(lines)
    hdr = f"[api:{file_rel}]"
    return hdr, body


__all__ = [
    "find_endpoints",
    "find_models",
    "extract_api_edges",
]
