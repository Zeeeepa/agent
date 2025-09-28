from __future__ import annotations

from typing import Any, Dict, Optional

from jinx.micro.llm.service import spark_openai
from jinx.micro.llm.chain_utils import truthy_env, extract_tagged_block, parse_planner_block
from jinx.micro.llm.chain_evidence import collect_pre_evidence
from jinx.micro.llm.chain_trace import trace_plan
from jinx.micro.llm.chain_resilience import record_success, record_failure, save_last_plan


async def run_planner(user_text: str, *, max_subqueries: Optional[int] = None, planner_ms: Optional[int] = 400) -> Dict[str, Any]:
    """Run a minimal planning step to produce a few short sub-queries.

    Returns a dict like: {"sub_queries": [...], "note": "..."}
    At most one LLM call; gated by JINX_CHAINED_REASONING.
    """
    # Planner is always enabled; keep running for any non-empty input

    txt = (user_text or "").strip()
    if not txt:
        return {"sub_queries": [], "note": "empty"}

    evid = await collect_pre_evidence(txt)
    planner_input = txt if not evid else (f"<user>\n{txt}\n</user>\n\n<evidence>\n{evid}\n</evidence>")
    # Continuity: inject a tiny continuity block (anchors) to stabilize planning
    try:
        from jinx.micro.conversation.cont import load_last_anchors as _load_last_anchors
        anc = await _load_last_anchors()
    except Exception:
        anc = {}
    try:
        lines = []
        q = (anc.get("questions") or [])[:1]
        if q:
            lines.append(f"q: {q[0]}")
        sy = (anc.get("symbols") or [])[:3]
        if sy:
            lines.append("symbols: " + ", ".join(sy))
        pth = (anc.get("paths") or [])[:2]
        if pth:
            lines.append("paths: " + ", ".join(pth))
        cont_block = ("<continuity>\n" + "\n".join(lines) + "\n</continuity>") if lines else ""
    except Exception:
        cont_block = ""
    if cont_block:
        planner_input = (planner_input + "\n\n" + cont_block) if planner_input else cont_block

    await trace_plan({"phase": "pre", "has_evidence": bool(evid)})
    try:
        # Advisory mode (soft influence) vs directive (legacy plan.*). Default: advisory.
        prompt_name = "planner_advisoryjson" if truthy_env("JINX_CHAINED_ADVISORY", "1") else "planner_minjson"
        out, tag = await spark_openai(planner_input, prompt_override=prompt_name)
    except Exception:
        await trace_plan({"phase": "plan", "error": "openai_error"})
        await record_failure("openai_error")
        return {"goal": "", "plan": [], "sub_queries": [], "risks": [], "note": "openai_error"}

    block = extract_tagged_block(out, tag, "machine")
    if not block:
        await trace_plan({"phase": "plan", "error": "no_machine_block"})
        await record_failure("no_machine_block")
        return {"goal": "", "plan": [], "sub_queries": [], "risks": [], "note": "no_machine"}

    data = parse_planner_block(block)
    # Optional: extract reusable helper kernels
    try:
        kernels_code = extract_tagged_block(out, tag, "plan_kernels")
    except Exception:
        kernels_code = ""
    if kernels_code:
        data["kernels"] = kernels_code
    # Success path: record and persist last good plan
    try:
        await record_success()
        await save_last_plan(data)
    except Exception:
        pass
    await trace_plan({
        "phase": "plan",
        "subs": data.get("sub_queries", []),
        "plan_len": len(data.get("plan", [])),
        "risks_len": len(data.get("risks", [])),
    })
    return data
