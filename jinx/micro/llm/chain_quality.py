from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_plan(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate basic structural quality of a planner output.

    Returns (plan, warnings). Plan is returned unchanged (for future transforms).
    Warnings are human-readable, compact lines for optional <plan_warnings>.
    """
    warnings: List[str] = []
    goal = str(plan.get("goal") or "").strip()
    steps = plan.get("plan") or []
    subs = plan.get("sub_queries") or []
    risks = plan.get("risks") or []

    # Goal check
    if not goal:
        warnings.append("goal: missing")

    # Steps checks
    if not isinstance(steps, list) or not steps:
        warnings.append("plan: missing steps")
    else:
        for i, st in enumerate(steps, start=1):
            if not isinstance(st, dict):
                warnings.append(f"plan.{i}: not a dict")
                continue
            s = str(st.get("step") or "").strip()
            w = str(st.get("why") or "").strip()
            c = str(st.get("criteria") or "").strip()
            if not s:
                warnings.append(f"plan.{i}.step: missing")
            if not w:
                warnings.append(f"plan.{i}.why: missing")
            if not c:
                warnings.append(f"plan.{i}.criteria: missing")

    # Sub-queries checks
    if not isinstance(subs, list):
        warnings.append("sub: invalid type")
    else:
        if len(subs) == 0:
            warnings.append("sub: none provided (consider adding focused probes)")

    # Risks checks
    if not isinstance(risks, list):
        warnings.append("risk: invalid type")
    else:
        if len(risks) == 0:
            warnings.append("risk: none provided (identify invalidating traps)")

    # Note is optional; no strict checks

    return plan, warnings
