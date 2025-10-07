from __future__ import annotations
import os
from typing import Any, Dict, List

from jinx.micro.llm.chain_utils import truthy_env
from jinx.micro.llm.chain_evidence import gather_planner_evidence
from jinx.micro.llm.chain_reflect import run_reflector
from jinx.micro.llm.chain_trace import trace_plan
from jinx.micro.llm.chain_gate import should_run_planner
from jinx.micro.llm.chain_citations import build_citation_block
from jinx.micro.llm.chain_quality import validate_plan
from jinx.micro.llm.chain_resilience import (
    allow_execution,
    load_last_plan,
)
from jinx.micro.llm.chain_metrics import build_metrics_block
from jinx.micro.llm.chain_plan import run_planner as _run_planner
from jinx.micro.llm.chain_render import (
    render_plan_brain,
    render_plan_guidance,
    render_plan_cortex,
    render_plan_warnings,
    render_reflection_block,
    render_plan_kernels,
)
from jinx.micro.llm.chain_context import gather_context_for_subs
from jinx.micro.llm.chain_finalize import finalize_context
from jinx.micro.llm.kernel_sanitizer import sanitize_kernels


async def run_planner(user_text: str, *, max_subqueries: int | None = None, planner_ms: int | None = 400) -> Dict[str, Any]:
    """Delegate to micro-module implementation for planner call."""
    return await _run_planner(user_text, max_subqueries=max_subqueries, planner_ms=planner_ms)


async def build_planner_context(user_text: str) -> str:
    """Build additional context using planner sub-queries with tight budgets.

    Keeps budgets small to limit latency and API usage. Returns a concatenated
    string of context blocks (may include multiple <embeddings_context> and
    <embeddings_code> sections), or an empty string if nothing found or disabled.
    """
    # Heuristic gate: only run the chain when the query is complex enough
    if not should_run_planner(user_text):
        await trace_plan({"phase": "gate", "allowed": False})
        return ""
    await trace_plan({"phase": "gate", "allowed": True})
    advisory = truthy_env("JINX_CHAINED_ADVISORY", "1")
    # Resilience gate: honor temporary disable windows
    try:
        allowed = await allow_execution()
    except Exception:
        allowed = True
    if not allowed:
        await trace_plan({"phase": "resilience_block", "allowed": False})
        # Try last known good plan as a fallback brain
        try:
            last = await load_last_plan()
        except Exception:
            last = None
        if not last:
            return ""
        # Build brain/cortex/warnings from last plan
        parts: List[str] = []
        warns: List[str] = []
        if not advisory:
            try:
                _, warns = validate_plan(last)
            except Exception:
                warns = []
        try:
            brain = render_plan_guidance(last) if advisory else render_plan_brain(last)
            if brain:
                parts.append(brain)
        except Exception:
            pass
        try:
            if truthy_env("JINX_CHAINED_INCLUDE_CORTEX", "1"):
                cortex_block = render_plan_cortex(last)
                if cortex_block:
                    parts.append(cortex_block)
        except Exception:
            pass
        if (not advisory) and warns and truthy_env("JINX_CHAINED_INCLUDE_WARNINGS", "1"):
            try:
                parts.append(render_plan_warnings(warns))
            except Exception:
                pass
        final_ctx = "\n".join([p for p in parts if p])
        await finalize_context(user_text, last, final_ctx)
        return final_ctx
    plan = await run_planner(user_text)
    # Validate structural quality and render optional warnings block
    warns: List[str] = []
    if not advisory:
        try:
            plan, warns = validate_plan(plan)
        except Exception:
            warns = []
    # If the plan is effectively empty, attempt to fallback to last known good plan
    try:
        is_empty = not (str(plan.get("goal") or "").strip() or (plan.get("plan") or []) or (plan.get("sub_queries") or []) or (plan.get("risks") or []) or str(plan.get("note") or "").strip())
    except Exception:
        is_empty = False
    if is_empty:
        await trace_plan({"phase": "fallback", "reason": "empty_plan"})
        try:
            last = await load_last_plan()
        except Exception:
            last = None
        if last:
            plan = last
            try:
                plan, lwarns = validate_plan(plan)
                warns = (warns or []) + (lwarns or [])
            except Exception:
                pass
    subs: List[str] = plan.get("sub_queries") or []
    if advisory and truthy_env("JINX_CHAINED_CLARIFY_AS_SUBS", "1"):
        try:
            subs = subs + [str(x).strip() for x in (plan.get("clarifiers") or []) if str(x).strip()]
        except Exception:
            pass
    if not subs:
        # Even without subs, return the brain/warnings blocks if present
        parts: List[str] = []
        try:
            brain = render_plan_guidance(plan) if advisory else render_plan_brain(plan)
            if brain:
                parts.append(brain)
        except Exception:
            pass
        # Optional plan cortex block (persona hints, ignored by downstream if unknown)
        try:
            if truthy_env("JINX_CHAINED_INCLUDE_CORTEX", "1"):
                cortex_block = render_plan_cortex(plan)
                if cortex_block:
                    parts.append(cortex_block)
        except Exception:
            pass
        if warns and truthy_env("JINX_CHAINED_INCLUDE_WARNINGS", "1"):
            try:
                parts.append(render_plan_warnings(warns))
            except Exception:
                pass
        final_ctx = "\n".join([p for p in parts if p])
        await finalize_context(user_text, plan, final_ctx)
        return final_ctx

    # Tight budgets for extra retrieval
    dialog_ms = int(os.getenv("JINX_CHAINED_DIALOG_CTX_MS", "140"))
    proj_ms = int(os.getenv("JINX_CHAINED_PROJECT_CTX_MS", "500"))

    parts: List[str] = await gather_context_for_subs(subs, dialog_ms, proj_ms)
    # If planner provided reusable helper kernels, include them for the main brain
    try:
        kernels_code = str(plan.get("kernels") or "")
    except Exception:
        kernels_code = ""
    if kernels_code:
        try:
            safe_k = sanitize_kernels(kernels_code)
            if safe_k:
                kblock = render_plan_kernels(safe_k)
                if kblock:
                    parts.append(kblock)
        except Exception:
            pass

    # Summarize the plan itself as a compact block to guide the final reasoning
    try:
        brain = render_plan_guidance(plan) if advisory else render_plan_brain(plan)
        if brain:
            parts.append(brain)
    except Exception:
        pass
    # Optional plan cortex block (persona hints)
    try:
        if truthy_env("JINX_CHAINED_INCLUDE_CORTEX", "1"):
            cortex_block = render_plan_cortex(plan)
            if cortex_block:
                parts.append(cortex_block)
    except Exception:
        pass
    # Append plan quality warnings if any
    if (not advisory) and warns and truthy_env("JINX_CHAINED_INCLUDE_WARNINGS", "1"):
        try:
            parts.append(render_plan_warnings(warns))
        except Exception:
            pass

    # Optional reflection: prefer using the combined prompt output to avoid a second API call
    try:
        evidence = await gather_planner_evidence(subs)
        # Trace aggregate evidence sizes
        await trace_plan({
            "phase": "evidence",
            "q_count": len(subs),
            "d_total": sum(len(x.get("dialogue", [])) for x in evidence.get("queries", [])),
            "c_total": sum(len(x.get("code", [])) for x in evidence.get("queries", [])),
        })
        try:
            reflect = dict(plan.get("reflect") or {})
        except Exception:
            reflect = {}
        if not reflect:
            reflect = await run_reflector(user_text, plan, evidence)
    except Exception:
        reflect = {}
        evidence = {}
    ref_block = ""
    if reflect:
        ref_block = render_reflection_block(reflect)
    if ref_block:
        parts.append(ref_block)
    # If reflector provided helper kernels for next steps, include them as well
    try:
        rk = str(reflect.get("kernels") or "") if reflect else ""
    except Exception:
        rk = ""
    if rk:
        try:
            safe_rk = sanitize_kernels(rk)
            if safe_rk:
                rkb = render_plan_kernels(safe_rk)
                if rkb:
                    parts.append(rkb)
        except Exception:
            pass
    # Optional compact citations block from gathered evidence
    try:
        if truthy_env("JINX_CHAINED_INCLUDE_CITATIONS", "1"):
            cit = build_citation_block(evidence)
            if cit:
                parts.append(cit)
    except Exception:
        pass
    # Optional metrics block for observability
    try:
        if truthy_env("JINX_CHAINED_INCLUDE_METRICS", "0"):
            met = build_metrics_block(evidence, plan)
            if met:
                parts.append(met)
    except Exception:
        pass
    final_ctx = "\n".join([p for p in parts if p])
    await finalize_context(user_text, plan, final_ctx)
    return final_ctx
