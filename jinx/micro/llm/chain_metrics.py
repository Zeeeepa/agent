from __future__ import annotations

from typing import Any, Dict, List


def _avg(scores: List[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def build_metrics_block(evidence: Dict[str, Any], plan: Dict[str, Any]) -> str:
    """Render a compact metrics block for the planner chain.

    Returns a string like:
    <plan_metrics>
    plan.steps: 3
    plan.subs: 2
    plan.risks: 2
    evidence.queries: 2
    evidence.dialogue.total: 3
    evidence.dialogue.avg: 0.74
    evidence.code.total: 3
    evidence.code.avg: 0.68
    </plan_metrics>
    """
    if not isinstance(evidence, dict):
        return ""
    queries = evidence.get("queries") or []
    d_scores: List[float] = []
    c_scores: List[float] = []
    d_total = 0
    c_total = 0
    for q in queries:
        for rec in (q.get("dialogue") or []):
            try:
                d_scores.append(float(rec.get("score") or 0.0))
                d_total += 1
            except Exception:
                continue
        for rec in (q.get("code") or []):
            try:
                c_scores.append(float(rec.get("score") or 0.0))
                c_total += 1
            except Exception:
                continue
    steps = plan.get("plan") or []
    subs = plan.get("sub_queries") or []
    risks = plan.get("risks") or []
    lines = [
        f"plan.steps: {len(steps) if isinstance(steps, list) else 0}",
        f"plan.subs: {len(subs) if isinstance(subs, list) else 0}",
        f"plan.risks: {len(risks) if isinstance(risks, list) else 0}",
        f"evidence.queries: {len(queries)}",
        f"evidence.dialogue.total: {d_total}",
        f"evidence.dialogue.avg: {_avg(d_scores):.2f}",
        f"evidence.code.total: {c_total}",
        f"evidence.code.avg: {_avg(c_scores):.2f}",
    ]
    return "<plan_metrics>\n" + "\n".join(lines) + "\n</plan_metrics>"
