from __future__ import annotations

from typing import Any, Dict, List


def render_plan_brain(plan: Dict[str, Any]) -> str:
    lines: List[str] = []
    g = str(plan.get("goal") or "").strip()
    if g:
        lines.append(f"goal: {g}")
    steps = plan.get("plan") or []
    for idx, st in enumerate(steps, start=1):
        if not isinstance(st, dict):
            continue
        s = str(st.get("step") or "").strip()
        w = str(st.get("why") or "").strip()
        c = str(st.get("criteria") or "").strip()
        if s:
            lines.append(f"plan.{idx}.step: {s}")
        if w:
            lines.append(f"plan.{idx}.why: {w}")
        if c:
            lines.append(f"plan.{idx}.criteria: {c}")
    subs_norm = [str(x).strip() for x in (plan.get("sub_queries") or []) if str(x).strip()]
    for j, sq in enumerate(subs_norm, start=1):
        lines.append(f"sub.{j}: {sq}")
    risks = [str(x).strip() for x in (plan.get("risks") or []) if str(x).strip()]
    for rj, rk in enumerate(risks, start=1):
        lines.append(f"risk.{rj}: {rk}")
    nt = str(plan.get("note") or "").strip()
    if nt:
        lines.append(f"note: {nt}")
    if not lines:
        return ""
    return "<plan_brain>\n" + "\n".join(lines) + "\n</plan_brain>"


def render_plan_cortex(plan: Dict[str, Any]) -> str:
    cortex = plan.get("cortex") or {}
    if not isinstance(cortex, dict) or not cortex:
        return ""
    clines: List[str] = []
    for ck, cv in cortex.items():
        if not cv:
            continue
        clines.append(f"cortex.{ck}: {str(cv)}")
    if not clines:
        return ""
    return "<plan_cortex>\n" + "\n".join(clines) + "\n</plan_cortex>"


def render_plan_warnings(warns: List[str]) -> str:
    if not warns:
        return ""
    return "<plan_warnings>\n" + "\n".join(f"- {w}" for w in warns) + "\n</plan_warnings>"


def render_reflection_block(reflect: Dict[str, Any]) -> str:
    if not reflect:
        return ""
    lines: List[str] = []
    if reflect.get("summary"):
        lines.append(reflect["summary"]) 
    items = reflect.get("next_actions") or []
    if items:
        label = "Nudges" if str(reflect.get("mode") or "").lower() == "advisory" else "Next"
        lines.append(f"{label}:\n- " + "\n- ".join(items))
    if not lines:
        return ""
    return "<plan_reflection>\n" + "\n".join(lines) + "\n</plan_reflection>"


def render_plan_guidance(plan: Dict[str, Any]) -> str:
    """Render an advisory guidance block that does NOT prescribe actions.

    Uses fields parsed from advisory planner:
      - goal (as need)
      - advice_do, advice_avoid
      - clarifiers, reminders, assumptions, context
    """
    lines: List[str] = []
    need = str(plan.get("goal") or "").strip()
    if need:
        lines.append(f"need: {need}")
    def _emit_list(items: List[str] | None, prefix: str) -> None:
        arr = [str(x).strip() for x in (items or []) if str(x).strip()]
        for i, v in enumerate(arr, start=1):
            lines.append(f"{prefix}.{i}: {v}")
    _emit_list(plan.get("advice_do"), "do")
    _emit_list(plan.get("advice_avoid"), "avoid")
    _emit_list(plan.get("clarifiers"), "clarify")
    _emit_list(plan.get("reminders"), "remind")
    _emit_list(plan.get("assumptions"), "assume")
    _emit_list(plan.get("context"), "context")
    if not lines:
        return ""
    return "<plan_guidance>\n" + "\n".join(lines) + "\n</plan_guidance>"


def render_plan_kernels(code: str | None) -> str:
    """Render optional reusable Python helper kernels provided by planner/reflector.

    Kernels are advisory utilities (stdlib-only) intended to be reused by the main brain.
    """
    if not code:
        return ""
    body = str(code).strip()
    if not body:
        return ""
    return "<plan_kernels>\n" + body + "\n</plan_kernels>"
