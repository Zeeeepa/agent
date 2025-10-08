from __future__ import annotations

import traceback
from typing import Optional
import os
import re

from jinx.logging_service import glitch_pulse, bomb_log, blast_mem
from jinx.openai_service import spark_openai
from jinx.error_service import dec_pulse
from jinx.conversation import build_chains, run_blocks
from jinx.micro.ui.output import pretty_echo
from jinx.micro.conversation.sandbox_view import show_sandbox_tail
from jinx.micro.conversation.error_report import corrupt_report
from jinx.logger.file_logger import append_line as _log_append
from jinx.log_paths import BLUE_WHISPERS
from jinx.micro.embeddings.retrieval import build_context_for
from jinx.micro.embeddings.project_retrieval import build_project_context_for, build_project_context_multi_for
from jinx.micro.embeddings.pipeline import embed_text
from jinx.conversation.formatting import build_header, ensure_header_block_separation
from jinx.micro.memory.storage import read_evergreen
from jinx.micro.memory.storage import read_channel as _read_channel
from jinx.micro.conversation.memory_sanitize import sanitize_transcript_for_memory
from jinx.micro.embeddings.project_config import ENABLE as PROJ_EMB_ENABLE
from jinx.micro.embeddings.project_paths import PROJECT_FILES_DIR
from jinx.micro.llm.chains import build_planner_context
from jinx.micro.llm.chain_persist import persist_memory
from jinx.micro.llm.kernel_sanitizer import sanitize_kernels as _sanitize_kernels
from jinx.micro.exec.executor import spike_exec as _spike_exec
from jinx.safety import chaos_taboo as _chaos_taboo
from jinx.micro.runtime.patcher import ensure_patcher_running as _ensure_patcher
from jinx.micro.conversation.cont import (
    augment_query_for_retrieval as _augment_query,
    maybe_reuse_last_context as _reuse_proj_ctx,
    save_last_context as _save_proj_ctx,
    extract_anchors as _extract_anchors,
    load_last_anchors as _load_last_anchors,
    render_continuity_block as _render_cont_block,
    last_agent_question as _last_q,
    last_user_query as _last_u,
    is_short_followup as _is_short,
    detect_topic_shift as _topic_shift,
    maybe_compact_state_frames as _compact_frames,
)
from jinx.micro.conversation.cont.classify import find_semantic_question as _find_semq
from jinx.micro.conversation.state_frame import build_state_frame
from jinx.micro.memory.router import assemble_memroute as _memroute


async def shatter(x: str, err: Optional[str] = None) -> None:
    """Drive a single conversation step and optionally handle an error context."""
    try:
        # Ensure micro-program runtime and event bridge are active before any code execution
        try:
            await _ensure_runtime()
            # Ensure the background AutoPatchProgram is running so model code can submit edits
            try:
                await _ensure_patcher()
            except Exception:
                pass
            # Ensure the embedding-based verifier is running for post-commit checks
            try:
                await _ensure_verifier()
            except Exception:
                pass
        except Exception:
            pass
        # Append the user input to the transcript first to ensure ordering
        if x and x.strip():
            await blast_mem(f"User: {x.strip()}")
            # Also embed the raw user input for retrieval (source: dialogue)
            try:
                await embed_text(x.strip(), source="dialogue", kind="user")
            except Exception:
                pass
        synth = await glitch_pulse()
        # Do not include the transcript in 'chains' since it is placed into <memory>
        # Do not inject error text into the body chains; it will live in <error>
        chains, decay = build_chains("", None)
        # Build standardized header blocks in a stable order before the main chains
        # 1) <embeddings_context> from recent dialogue/sandbox using current input as query,
        #    plus project code embeddings context assembled from emb/ when available
        # Continuity: augment retrieval query on short clarifications
        topic_shifted = False
        reuse_for_log = False
        try:
            q_raw = (x or synth or "")
            continuity_on = str(os.getenv("JINX_CONTINUITY_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
            anchors = {}
            if continuity_on:
                try:
                    cur = _extract_anchors(synth or "")
                except Exception:
                    cur = {}
                # Optional: boost with semantic question detector (language-agnostic)
                try:
                    semq = await _find_semq(synth or "")
                    if semq:
                        qs = [semq]
                        for qline in (cur.get("questions") or []):
                            if qline != semq:
                                qs.append(qline)
                        cur["questions"] = qs
                except Exception:
                    pass
                try:
                    prev = await _load_last_anchors()
                except Exception:
                    prev = {}
                # merge anchors (current first, then previous uniques), cap lists
                anchors = {k: list(dict.fromkeys((cur.get(k) or []) + (prev.get(k) or [])))[:10] for k in set((cur or {}).keys()) | set((prev or {}).keys())}
                eff_q = _augment_query(x or "", synth or "", anchors=anchors)
            else:
                eff_q = q_raw
            base_ctx = await build_context_for(eff_q)
        except Exception:
            base_ctx = ""
        # Always build project context; retrieval enforces its own tight budgets
        proj_ctx = ""
        try:
            _q = eff_q
            # Augment the project query with routed memory hints to bind retrieval to the user's current intent
            try:
                mem_hints = await _memroute(_q, k=8, preview_chars=120)
            except Exception:
                mem_hints = []
            try:
                max_q_chars = int(os.getenv("JINX_PROJ_QUERY_MAX_CHARS", "800"))
            except Exception:
                max_q_chars = 800
            _q_proj = (" ".join([_q] + mem_hints)).strip()
            if len(_q_proj) > max_q_chars:
                _q_proj = _q_proj[:max_q_chars]
            qlow = _q_proj.lower()
            codey = any(sym in _q_proj for sym in "=[](){}.:,") or any(kw in qlow for kw in ["def ", "class ", "import ", "from ", "return ", "async ", "await ", " for ", " in ", " = "])
            # If code-like, give a bit more budget on first attempt
            first_budget = 1200 if codey else None
            # Dynamic top-K for project hits (higher for code-like queries)
            try:
                proj_k = int(os.getenv("JINX_PROJ_CTX_K", ("10" if codey else "6")))
            except Exception:
                proj_k = 10 if codey else 6
            # Build multiple enriched sub-queries (task fused with individual memory hints)
            try:
                subq_cap = int(os.getenv("JINX_PROJ_SUBQ_MAX", "3"))
            except Exception:
                subq_cap = 3
            subqs = []
            subqs.append(_q_proj)
            # 1) from memory hints (routed memory lines)
            for h in (mem_hints or [])[:max(0, subq_cap)]:
                qh = f"{_q} {h}".strip()
                if len(qh) > max_q_chars:
                    qh = qh[:max_q_chars]
                if qh not in subqs:
                    subqs.append(qh)
            # 2) from channels: top paths and symbols (if available)
            try:
                ch_paths = (await _read_channel("paths") or "").splitlines()
            except Exception:
                ch_paths = []
            try:
                ch_syms = (await _read_channel("symbols") or "").splitlines()
            except Exception:
                ch_syms = []
            def _extract_after(prefix: str, ln: str) -> str:
                low = (ln or "").lower()
                if low.startswith(prefix):
                    return ln[len(prefix):].strip()
                return ""
            add_more = max(0, subq_cap - max(0, len(subqs) - 1))
            # Prefer a path and a symbol if room allows
            for ln in ch_paths[:2]:
                if add_more <= 0:
                    break
                p = _extract_after("path: ", ln)
                if not p:
                    continue
                # Skip internal memory paths like .jinx/**
                try:
                    pp = p.replace("\\", "/").lower()
                    if any(seg == ".jinx" for seg in pp.split("/")):
                        continue
                except Exception:
                    pass
                qh = f"{_q} {p}".strip()
                if len(qh) > max_q_chars:
                    qh = qh[:max_q_chars]
                if qh not in subqs:
                    subqs.append(qh)
                    add_more -= 1
            for ln in ch_syms[:2]:
                if add_more <= 0:
                    break
                s = _extract_after("symbol: ", ln)
                if not s:
                    continue
                qh = f"{_q} {s}".strip()
                if len(qh) > max_q_chars:
                    qh = qh[:max_q_chars]
                if qh not in subqs:
                    subqs.append(qh)
                    add_more -= 1
            # Prefer multi-query aggregation when we have hints; fallback to single-query
            if len(subqs) > 1:
                proj_ctx = await build_project_context_multi_for(subqs, k=proj_k, max_time_ms=first_budget)
            else:
                proj_ctx = await build_project_context_for(_q_proj, k=proj_k, max_time_ms=first_budget)
            if not proj_ctx:
                # Cold-start fallback: retry with a larger time budget
                if len(subqs) > 1:
                    proj_ctx = await build_project_context_multi_for(subqs, k=proj_k, max_time_ms=2000)
                if not proj_ctx:
                    proj_ctx = await build_project_context_for(_q_proj, k=proj_k, max_time_ms=2000)
            # Continuity: if still empty and this is a short clarification, reuse last cached project context
            if not proj_ctx:
                reuse = ""
                try:
                    ts_check = str(os.getenv("JINX_TOPIC_SHIFT_CHECK", "1")).lower() not in ("", "0", "false", "off", "no")
                    if ts_check and _is_short(x or ""):
                        shifted = await _topic_shift(_q)
                        topic_shifted = topic_shifted or bool(shifted)
                        if not shifted:
                            reuse = await _reuse_proj_ctx(x or "", proj_ctx, synth or "")
                    else:
                        reuse = await _reuse_proj_ctx(x or "", proj_ctx, synth or "")
                except Exception:
                    reuse = ""
                if reuse:
                    proj_ctx = reuse
                    reuse_for_log = True
        except Exception:
            proj_ctx = ""
        # Persist last project context snapshot for continuity cache
        try:
            await _save_proj_ctx(proj_ctx or "", anchors=anchors if 'anchors' in locals() else None)
        except Exception:
            pass
        # Optional: planner-enhanced context (adds at most one extra LLM call + small retrieval)
        plan_ctx = ""
        try:
            if str(os.getenv("JINX_PLANNER_CTX", "0")).lower() not in ("", "0", "false", "off", "no"):
                plan_ctx = await build_planner_context(_q)
        except Exception:
            plan_ctx = ""
        # Optional continuity block for the main brain
        try:
            cont_block = _render_cont_block(
                anchors if 'anchors' in locals() else None,
                _last_q(synth or ""),
                _last_u(synth or ""),
                _is_short(x or ""),
            )
        except Exception:
            cont_block = ""
        ctx = "\n".join([c for c in [base_ctx, proj_ctx, plan_ctx, cont_block] if c])

        # Optional: Preload sanitized <plan_kernels> code before final execution, guarded by env
        try:
            if str(os.getenv("JINX_KERNELS_PRELOAD", "0")).lower() not in ("", "0", "false", "off", "no") and plan_ctx:
                pk = []
                s = plan_ctx
                pos = 0
                ltag = "<plan_kernels>"; rtag = "</plan_kernels>"
                while True:
                    i = s.find(ltag, pos)
                    if i == -1:
                        break
                    j = s.find(rtag, i)
                    if j == -1:
                        break
                    body = s[i + len(ltag): j]
                    pos = j + len(rtag)
                    safe = _sanitize_kernels(body)
                    if safe:
                        pk.append(safe)
                if pk:
                    async def _preload_cb(err_msg):
                        if err_msg:
                            await bomb_log(f"kernel preload error: {err_msg}")
                    for code in pk:
                        try:
                            await _spike_exec(code, _chaos_taboo, _preload_cb)
                        except Exception:
                            pass
        except Exception:
            pass

        # Continuity: persist a compact state frame via embeddings for next turns
        try:
            if str(os.getenv("JINX_STATEFRAME_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no"):
                guid = plan_ctx or ""
                state_frame = build_state_frame(
                    user_text=(x or ""),
                    synth=synth or "",
                    anchors=anchors if 'anchors' in locals() else None,
                    guidance=guid,
                    cont_block=cont_block,
                    error_summary=(err.strip() if err and isinstance(err, str) else ""),
                )
                if state_frame and state_frame.strip():
                    # Deduplicate by content hash to avoid drift/bloat
                    import hashlib as _hashlib
                    from jinx.micro.conversation.cont import load_cache_meta as _load_meta, save_last_context_with_meta as _save_meta
                    sha = _hashlib.sha256(state_frame.encode("utf-8", errors="ignore")).hexdigest()
                    try:
                        meta = await _load_meta()
                    except Exception:
                        meta = {}
                    if (meta.get("frame_sha") or "") != sha:
                        await embed_text(state_frame, source="state", kind="frame")
                        # Also update meta with the frame hash to gate future duplicates
                        try:
                            await _save_meta(proj_ctx or "", anchors if 'anchors' in locals() else None, frame_sha=sha)
                        except Exception:
                            pass
                # Attempt periodic concept compaction (fast no-op if not time)
                try:
                    await _compact_frames()
                except Exception:
                    pass
        except Exception:
            pass
        # 2) <memory> from transcript (exclude the latest user input line and sanitize)
        mem_text = sanitize_transcript_for_memory(synth or "", (x or "").strip())
        # 2.5) <evergreen> persistent durable facts
        try:
            evergreen_text = (await read_evergreen()) or ""
        except Exception:
            evergreen_text = ""
        # Continuity: optionally gate evergreen by topic shift on short follow-ups to avoid bleed
        try:
            if str(os.getenv("JINX_EVERGREEN_TOPIC_GUARD", "1")).lower() not in ("", "0", "false", "off", "no"):
                if _is_short(x or ""):
                    try:
                        shifted = await _topic_shift(_q)
                    except Exception:
                        shifted = False
                    topic_shifted = topic_shifted or bool(shifted)
                    if shifted:
                        evergreen_text = ""
        except Exception:
            pass
        # Optional: persist memory snapshot as Markdown for project embeddings ingestion
        try:
            if (os.getenv("JINX_PERSIST_MEMORY", "1").strip().lower() not in ("", "0", "false", "off", "no")):
                await persist_memory(mem_text, evergreen_text, user_text=(x or ""), plan_goal="")
        except Exception:
            pass
        # 3) <task> reflects the immediate objective: when handling an error,
        #    avoid copying traceback or transcript into <task>.
        #    Continuity augmentation disabled: use only the current user input.
        if err and err.strip():
            task_text = ""
        else:
            task_text = (x or "").strip()
        # Optional <error> block carries execution or prior error details
        error_text = (err.strip() if err and err.strip() else None)

        # Assemble header using shared formatting utilities
        header_text = build_header(ctx, mem_text, task_text, error_text, evergreen_text)
        if header_text:
            chains = header_text + ("\n\n" + chains if chains else "")
        # Continuity dev echo (optional): tiny trace line for observability
        try:
            if str(os.getenv("JINX_CONTINUITY_DEV_ECHO", "0")).lower() not in ("", "0", "false", "off", "no"):
                sym_n = len(anchors.get("symbols", [])) if 'anchors' in locals() else 0
                pth_n = len(anchors.get("paths", [])) if 'anchors' in locals() else 0
                await _log_append(BLUE_WHISPERS, f"[CONT] short={int(_is_short(x or ''))} topic_shift={int(topic_shifted)} reuse={int(reuse_for_log)} sym={sym_n} path={pth_n}")
        except Exception:
            pass
        # If an error is present, enforce a decay hit to drive auto-fix loop
        if err and err.strip():
            decay = max(decay, 50)
        if decay:
            await dec_pulse(decay)
        # Final normalization guard
        chains = ensure_header_block_separation(chains)
        # Use a dedicated recovery prompt only when fixing an error; otherwise default prompt
        prompt_override = "burning_logic_recovery" if (err and err.strip()) else None
        out, code_id = await spark_openai(chains, prompt_override=prompt_override)

        # Ensure that on any execution error we also show the raw model output
        async def on_exec_error(err_msg: Optional[str]) -> None:
            # Sandbox callback sends None on success — ignore to avoid duplicate log prints
            if not err_msg:
                return
            pretty_echo(out)
            await show_sandbox_tail()
            await corrupt_report(err_msg)

        executed = await run_blocks(out, code_id, on_exec_error)
        if not executed:
            await bomb_log(f"No executable <python_{code_id}> block found in model output; displaying raw output.")
            pretty_echo(out)
            await dec_pulse(10)
            # Log a clean Jinx line (prefer question content); avoid raw tags
            try:
                pairs = parse_tagged_blocks(out, code_id)
            except Exception:
                pairs = []
            qtext = ""
            for tag, core in pairs:
                if tag.startswith("python_question_"):
                    qtext = (core or "").strip()
                    break
            if not qtext:
                try:
                    txt = out or ""
                    txt = re.sub(r"<[^>]+>.*?</[^>]+>", "", txt, flags=re.DOTALL)
                    txt = re.sub(r"<[^>]+>", "", txt)
                    qtext = txt.strip()
                except Exception:
                    qtext = (out or "").strip()
            if qtext:
                await blast_mem(f"Jinx: {qtext}")
        else:
            # After successful execution, also surface the latest sandbox log context
            await show_sandbox_tail()
            # Also embed the agent output for retrieval (source: dialogue)
            try:
                await embed_text(out.strip(), source="dialogue", kind="agent")
            except Exception:
                pass
    except Exception:
        await bomb_log(traceback.format_exc())
        await dec_pulse(50)
    finally:
        # Run memory optimization after each model interaction using a per-turn snapshot
        snap = await glitch_pulse()
        # Late import avoids circular import during startup
        from jinx.micro.memory.optimizer import submit as _opt_submit
        await _opt_submit(snap)
