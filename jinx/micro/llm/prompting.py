from __future__ import annotations

import os
import json
from typing import List, Tuple, Optional


def derive_basic_context(max_resources: int = 4) -> Tuple[str, List[str]]:
    """Infer basic project name and candidate resource names without user input.
    - name: from README title or folder name
    - resources: top-level directory names filtered by heuristics
    """
    # Project name
    try:
        root = os.getcwd()
        name = os.path.basename(root).strip().lower().replace(" ", "_") or "api"
        rd = os.path.join(root, "README.md")
        if os.path.isfile(rd):
            try:
                with open(rd, "r", encoding="utf-8", errors="ignore") as f:
                    head = "\n".join([next(f, "") for _ in range(5)])
                for line in head.splitlines():
                    s = line.strip("# ")
                    if 2 <= len(s) <= 64:
                        name = s.strip().lower().replace(" ", "_")
                        break
            except Exception:
                pass
    except Exception:
        name = "api"
    # Candidate resources
    res: List[str] = []
    try:
        ignore = {"api", ".git", ".jinx", "jinx", "venv", ".venv", "node_modules", "__pycache__"}
        for d in os.listdir(os.getcwd()):
            dn = d.strip().lower()
            if dn in ignore:
                continue
            p = os.path.join(os.getcwd(), d)
            if os.path.isdir(p) and 3 <= len(dn) <= 24:
                res.append(dn)
            if len(res) >= max_resources:
                break
    except Exception:
        pass
    return name or "api", res


def build_api_spec_prompt(
    request: Optional[str],
    project_name: str,
    candidate_resources: List[str],
) -> str:
    """Construct a strict prompt for API spec synthesis.
    Returns a compact instruction that enforces a JSON-only response.
    """
    req = (request or "Generate a pragmatic REST API for this repository.").strip()
    shape = (
        '{\n  "name": "string",\n  "resources": [\n    {\n      "name": "string",\n      "fields": {"id": "int|str|float|bool", "...": "..."},\n      "endpoints": ["list", "get", "create", "update", "delete"]\n    }\n  ]\n}'
    )
    prompt = (
        "You are an autonomous system planner. Produce ONLY a JSON object for a REST API spec.\n"
        "Respond with ASCII only, no code fences, no commentary. Use this exact shape:\n"
        f"{shape}\n\n"
        "Constraints: max 4 resources, max 6 fields per resource. Prefer pragmatic defaults.\n"
        f"Project name: {project_name}\n"
        f"Candidate resources: {json.dumps(candidate_resources)}\n"
        f"Request: {req}\n"
    )
    return prompt


__all__ = [
    "derive_basic_context",
    "build_api_spec_prompt",
]
