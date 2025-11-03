from __future__ import annotations

import asyncio
from typing import Optional, Tuple
import time
from jinx.micro.common.log import log_info as _log_info

from jinx.micro.rt.rt_budget import run_bounded as _bounded_run, env_ms as _env_ms
from jinx.micro.rt.activity import set_activity_detail as _actdet
from jinx.micro.embeddings.retrieval import build_context_for as _build_base_ctx
from jinx.micro.embeddings.memory_context import build_memory_context_for as _build_mem_ctx
from jinx.micro.conversation.proj_context_enricher import build_project_context_enriched as _build_proj_ctx_enriched
from jinx.micro.llm.service import spark_openai as _spark_llm, spark_openai_streaming as _spark_llm_stream
from jinx.conversation import run_blocks as _run_blocks


_RT_SUMMARY: dict[str, dict[str, float | int | bool]] = {}


def _rec(stage: str, start: float, res: Optional[str], budget_ms_env: str, default_ms: int) -> None:
    dur_ms = int((time.perf_counter() - start) * 1000.0)
    ms = _env_ms(budget_ms_env, default_ms)
    timed_out = (res is None and ms > 0)
    _RT_SUMMARY[stage] = {"dur_ms": dur_ms, "timed_out": timed_out}


async def build_runtime_base_ctx(query: str) -> str:
    try:
        _actdet({"stage": "base_ctx", "rem_ms": _env_ms("JINX_STAGE_BASECTX_MS", 220)})
    except Exception:
        pass
    t0 = time.perf_counter()
    res = await _bounded_run(_build_base_ctx(query), _env_ms("JINX_STAGE_BASECTX_MS", 220))
    _rec("base_ctx", t0, res, "JINX_STAGE_BASECTX_MS", 220)
    return res or ""


async def build_runtime_mem_ctx(query: str) -> str:
    try:
        _actdet({"stage": "mem_ctx", "rem_ms": _env_ms("JINX_STAGE_MEMCTX_MS", 160)})
    except Exception:
        pass
    t0 = time.perf_counter()
    res = await _bounded_run(_build_mem_ctx(query), _env_ms("JINX_STAGE_MEMCTX_MS", 160))
    _rec("mem_ctx", t0, res, "JINX_STAGE_MEMCTX_MS", 160)
    return res or ""


async def build_project_context_enriched(query: str, *, user_text: str, synth: str) -> str:
    try:
        _actdet({"stage": "proj_ctx", "rem_ms": _env_ms("JINX_STAGE_PROJCTX_MS", 260)})
    except Exception:
        pass
    t0 = time.perf_counter()
    res = await _bounded_run(_build_proj_ctx_enriched(query, user_text=user_text, synth=synth), _env_ms("JINX_STAGE_PROJCTX_MS", 260))
    _rec("proj_ctx", t0, res, "JINX_STAGE_PROJCTX_MS", 260)
    return res or ""


async def call_llm(chains: str, *, prompt_override: Optional[str], stream_on: bool, on_first_block) -> Tuple[str, str]:
    try:
        _actdet({"stage": "llm", "rem_ms": _env_ms("JINX_STAGE_LLM_MS", 45000)})
    except Exception:
        pass
    aw = (_spark_llm_stream(chains, prompt_override=prompt_override, on_first_block=on_first_block)
          if stream_on else _spark_llm(chains, prompt_override=prompt_override))
    t0 = time.perf_counter()
    res = await _bounded_run(aw, _env_ms("JINX_STAGE_LLM_MS", 45000))
    # res is Tuple[str, str] or None
    _rec("llm", t0, ("" if res is None else "ok"), "JINX_STAGE_LLM_MS", 45000)
    if res is None:
        # Fallback: perform an unbounded non-streaming call to avoid blank replies
        try:
            out2, tag2 = await _spark_llm(chains, prompt_override=prompt_override)
            return out2, tag2
        except Exception:
            # Last resort stub; ensures the pipeline continues deterministically
            return "<llm_timeout>", "main"
    return res


async def execute_blocks(out: str, code_id: str, on_exec_error) -> bool:
    try:
        _actdet({"stage": "exec", "rem_ms": _env_ms("JINX_STAGE_EXEC_MS", 30000)})
    except Exception:
        pass
    t0 = time.perf_counter()
    res = await _bounded_run(_run_blocks(out, code_id, on_exec_error), _env_ms("JINX_STAGE_EXEC_MS", 30000))
    _rec("exec", t0, ("" if res is None else "ok"), "JINX_STAGE_EXEC_MS", 30000)
    ok = bool(res) if res is not None else False
    # Emit concise per-turn RT summary
    try:
        b = _RT_SUMMARY.get("base_ctx", {})
        m = _RT_SUMMARY.get("mem_ctx", {})
        p = _RT_SUMMARY.get("proj_ctx", {})
        l = _RT_SUMMARY.get("llm", {})
        e = _RT_SUMMARY.get("exec", {})
        _log_info(
            "rt.summary",
            base=int(b.get("dur_ms", 0) or 0),
            mem=int(m.get("dur_ms", 0) or 0),
            proj=int(p.get("dur_ms", 0) or 0),
            llm=int(l.get("dur_ms", 0) or 0),
            exec=int(e.get("dur_ms", 0) or 0),
            to=int(b.get("timed_out", False) or 0) + int(m.get("timed_out", False) or 0) + int(p.get("timed_out", False) or 0) + int(l.get("timed_out", False) or 0) + int(e.get("timed_out", False) or 0),
        )
    except Exception:
        pass
    finally:
        try:
            _RT_SUMMARY.clear()
        except Exception:
            pass
    return ok
