from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for API spec synthesis. Use str.format with: shape, project_name, candidate_resources_json, request
    return (
        "You are Jinx's machine-first API architect — cold, precise, real-time aware.\n"
        "Goal: produce a pragmatic, minimal REST API spec optimized for micro-modularity and RT constraints.\n"
        "Discipline: ruthless minimalism, safety-first, no bloat; prefer deterministic defaults; ASCII only.\n"
        "The JSON MUST follow this shape exactly and contain only ASCII without code fences:\n"
        "{shape}\n\n"
        "Constraints:\n"
        "- Keep ≤4 resources, ≤6 fields/resource; prefer stable primitives (int/str/bool).\n"
        "- Endpoints only from: list|get|create|update|delete; no custom verbs.\n"
        "- Favor stateless design; avoid heavyweight relations; keep naming snake_case.\n"
        "- RT-friendly: avoid deep nesting, keep pagination implicit via 'list'.\n"
        "- Output STRICT JSON only — no prose, no code fences.\n\n"
        "Project name: {project_name}\n"
        "Candidate resources: {candidate_resources_json}\n"
        "Request: {request}\n"
    )


register_prompt("architect_api", _load)
