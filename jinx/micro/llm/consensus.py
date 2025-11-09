from __future__ import annotations

import asyncio as _asyncio
import os
import re as _re
from typing import Callable

from .openai_caller import call_openai as _call_once


def _score(text: str) -> float:
    """Heuristic score for outputs. Prefer structured, codeful, and concise content."""
    if not text:
        return 0.0
    s = text.strip()
    score = 0.0
    n = len(s)
    # Prefer presence of fenced code or <python_*> blocks
    if "```" in s:
        score += 1.5
    if "<python_" in s and "</python_" in s:
        score += 1.0
    # Penalize overly long blobs
    if n <= 6000:
        score += 0.5
    # Simple Python parse attempt on first code fence
    try:
        import re as _re, ast as _ast
        m = _re.search(r"```(?:python)?\n([\s\S]*?)\n```", s)
        if m:
            body = m.group(1)
            _ast.parse(body)
            score += 1.0
    except Exception:
        pass
    return score


async def refine_output(instructions: str, model: str, input_text: str, base_text: str) -> str:
    """Try a tiny consensus: request one alternative candidate with a small variation
    and pick the higher-scoring output. Budgeted via env `JINX_LLM_CONSENSUS_MS`.
    If disabled or on error, returns base_text.
    """
    try:
        if str(os.getenv("JINX_LLM_CONSENSUS", "1")).lower() in ("", "0", "false", "off", "no"):
            return base_text
    except Exception:
        return base_text
    try:
        budget_ms = int(os.getenv("JINX_LLM_CONSENSUS_MS", "500"))
    except Exception:
        budget_ms = 500
    try:
        k = max(1, int(os.getenv("JINX_LLM_CONSENSUS_K", "3")))
    except Exception:
        k = 3
    # Skip alternates entirely if base already looks strong
    try:
        min_score = float(os.getenv("JINX_LLM_CONSENSUS_MIN_SCORE", "2.0"))
    except Exception:
        min_score = 2.0
    if _score(base_text or "") >= min_score:
        return base_text

    # Small prompt variation: append an instruction for determinism and structure (from prompts)
    try:
        from jinx.prompts import render_prompt as _render_prompt
        _alt = _render_prompt("consensus_alt")
    except Exception:
        _alt = "Return a precise, self-contained solution. Put final code in one fenced block."
    alt_inst = instructions + "\n\n" + _alt

    async def _one_alt(i: int) -> str:
        try:
            # Slightly vary the instruction for diversity
            inst = alt_inst + f"\nVariation:{i} — ensure determinism and correct code fences."
            return await _call_once(inst, model, input_text)
        except Exception:
            return ""

    # Generate K-1 alts concurrently, base is implicit candidate A
    alts: list[str] = []
    try:
        tasks = [_asyncio.create_task(_one_alt(i)) for i in range(1, k)]
        done, _ = await _asyncio.wait(tasks, timeout=max(0.05, budget_ms / 1000.0))
        for t in done:
            try:
                alts.append(t.result())
            except Exception:
                alts.append("")
    except Exception:
        alts = []
    # Include base and pick best by heuristic
    cand = [(base_text or "", "A0")]
    for i, txt in enumerate(alts, start=1):
        cand.append((txt or "", f"B{i}"))
    # Compute best by heuristic, and track second best for judge gating
    try:
        scored = [(_score(txt), tag, txt) for (txt, tag) in cand]
        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_tag, best_txt = (scored[0][0], scored[0][1], scored[0][2])
        second_score = scored[1][0] if len(scored) > 1 else 0.0
    except Exception:
        best_txt, best_tag = base_text, "A0"
        best_score, second_score = _score(base_text or ""), 0.0
    if best_tag.startswith("A"):
        best_other = cand[1][0] if len(cand) > 1 else base_text
    else:
        best_other = base_text

    # Optional LLM judge to break ties/confirm (skip when margin is large)
    try:
        if str(os.getenv("JINX_LLM_CONSENSUS_JUDGE", "1")).lower() in ("", "0", "false", "off", "no"):
            return best_txt
        judge_ms = int(os.getenv("JINX_LLM_CONSENSUS_JUDGE_MS", "450"))
    except Exception:
        return best_txt
    # If the best is significantly better than the rest, skip judge
    try:
        judge_min_delta = float(os.getenv("JINX_LLM_CONSENSUS_JUDGE_MIN_DELTA", "0.6"))
    except Exception:
        judge_min_delta = 0.6
    if (best_score - second_score) >= judge_min_delta:
        return best_txt

    # Sanitize candidates to avoid nested/duplicated [A]/[B] labels when passed into the judge
    def _sanitize_for_judge(s: str) -> str:
        t = (s or "").strip()
        if not t:
            return ""
        # Remove standalone [A]/[B] headers and inline markers
        t = _re.sub(r"(?mi)^\s*\[(?:A|B)\]\s*$", "", t)
        t = _re.sub(r"(?m)\n\[(?:A|B)\]\n", "\n", t)
        t = _re.sub(r"^\s*\[(?:A|B)\]\s*\n", "", t)
        return t.strip()
    try:
        judge_max = int(os.getenv("JINX_LLM_CONSENSUS_JUDGE_MAX", "4000"))
    except Exception:
        judge_max = 4000
    a_txt = _sanitize_for_judge(best_txt)[:judge_max]
    b_txt = _sanitize_for_judge(best_other)[:judge_max]
    try:
        from jinx.prompts import render_prompt as _render_prompt
        prompt = _render_prompt("consensus_judge", a=a_txt, b=b_txt)
    except Exception:
        prompt = ("[A]\n" + a_txt + "\n\n[B]\n" + b_txt)
    async def _judge() -> str:
        try:
            return await _call_once(prompt, model, "")
        except Exception:
            return ""
    try:
        jtxt = await _asyncio.wait_for(_judge(), timeout=max(0.05, judge_ms / 1000.0))
    except Exception:
        jtxt = ""
    pick = "A"
    if jtxt:
        try:
            import json as _json, re as _re
            m = _re.search(r"\{[\s\S]*\}", jtxt)
            obj = _json.loads(m.group(0) if m else jtxt)
            p = str(obj.get("pick") or "A").strip().upper()
            if p in ("A", "B"):
                pick = p
        except Exception:
            pick = "A"
    return best_other if pick == "B" else best_txt
