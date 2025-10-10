from __future__ import annotations

import traceback
from typing import Optional
import os
import re
import asyncio

from jinx.logging_service import glitch_pulse, bomb_log, blast_mem
from jinx.openai_service import spark_openai
from jinx.error_service import dec_pulse
from jinx.conversation import build_chains, run_blocks
from jinx.micro.ui.output import pretty_echo
from jinx.micro.conversation.sandbox_view import show_sandbox_tail
from jinx.micro.conversation.error_report import corrupt_report
from jinx.logger.file_logger import append_line as _log_append
from jinx.log_paths import BLUE_WHISPERS
from jinx.micro.recursor.normalizer import normalize_output_blocks
from jinx.micro.parser.api import parse_tagged_blocks
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
from jinx.micro.runtime.api import ensure_runtime as _ensure_runtime
from jinx.micro.verify.verifier import ensure_verifier_running as _ensure_verifier
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from jinx.micro.llm.service import spark_openai as _spark_llm, spark_openai_streaming as _spark_llm_stream
from jinx.micro.conversation.proj_context_enricher import build_project_context_enriched as _build_proj_ctx_enriched
from jinx.micro.conversation.error_payload import attach_error_code as _attach_error_code
from jinx.micro.memory.evergreen_select import select_evergreen_for as _select_evg
from jinx.micro.embeddings.memory_context import build_memory_context_for as _build_mem_ctx
from jinx.micro.memory.api_memory import build_api_memory_block as _build_api_mem, append_turn as _append_turn
from jinx.micro.conversation.turns_infer import infer_turn_query as _infer_turn
from jinx.micro.memory.turns import get_user_message as _turn_user, get_jinx_reply_to as _turn_jinx, parse_active_turns as _turns_all
from jinx.micro.conversation.memory_reasoner import infer_memory_action as _infer_memsel
from jinx.micro.memory.pin_store import load_pins as _pins_load
from jinx.micro.conversation.prefilter import likely_memory_action as _likely_mem
from jinx.micro.conversation.debug import log_debug
from jinx.micro.conversation.memory_program import infer_memory_program as _mem_program


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
            # Also embed the raw user input for retrieval (source: dialogue) in background
            try:
                asyncio.create_task(embed_text(x.strip(), source="dialogue", kind="user"))
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
            # Prefer error text for retrieval if present; fallback to user text, then transcript
            q_raw = (x or err or synth or "")
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
                eff_q = _augment_query((x or err or ""), synth or "", anchors=anchors)
            else:
                eff_q = q_raw
            # Launch runtime context retrieval concurrently with project context assembly
            base_ctx_task = asyncio.create_task(build_context_for(eff_q))
            # Optional: embeddings-backed memory context (no evergreen), env-gated (default OFF for API)
            mem_ctx_task = None
            try:
                if str(os.getenv("JINX_EMBED_MEMORY_CTX", "0")).lower() not in ("", "0", "false", "off", "no"):
                    mem_ctx_task = asyncio.create_task(_build_mem_ctx(eff_q))
            except Exception:
                mem_ctx_task = None
        except Exception:
            base_ctx_task = asyncio.create_task(asyncio.sleep(0.0))  # type: ignore
        # Always build project context; retrieval enforces its own tight budgets
        proj_ctx = ""
        try:
            _q = eff_q
            # Delegate enrichment/build to the dedicated micro-module (deduplicated logic)
            proj_ctx_task: asyncio.Task[str] = asyncio.create_task(_build_proj_ctx_enriched(_q, user_text=x or "", synth=synth or ""))
            # Await both contexts; runtime may already be done, otherwise this overlaps work
            try:
                base_ctx = await base_ctx_task
            except Exception:
                base_ctx = ""
            # Await memory context if launched (used internally only; not sent to API)
            mem_ctx = ""
            try:
                if 'mem_ctx_task' in locals() and mem_ctx_task is not None:
                    mem_ctx = await mem_ctx_task
            except Exception:
                mem_ctx = ""
            try:
                proj_ctx = await proj_ctx_task
            except Exception:
                proj_ctx = ""
            # _build_proj_ctx_enriched already implements its own fallback
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
            # If project assembly failed early, still await runtime context task
            try:
                base_ctx = await base_ctx_task
            except Exception:
                base_ctx = ""
            proj_ctx = ""
        # If runtime retrieval wasn't launched (fallback path), ensure base_ctx exists
        if 'base_ctx' not in locals():
            try:
                base_ctx = await build_context_for(eff_q)
            except Exception:
                base_ctx = ""
        if 'mem_ctx' not in locals():
            mem_ctx = ""
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
        # Optional: auto-turns resolver — hybrid (fast+LLM) that injects a tiny <turns> block when the user asks about Nth message
        turns_block = ""
        try:
            tq = await _infer_turn(x or "")
        except Exception:
            tq = None
        if tq:
            # Confidence gating to avoid false positives
            try:
                conf_min = float(os.getenv("JINX_TURNS_CONF_MIN", "0.3"))
            except Exception:
                conf_min = 0.3
            try:
                kind = (tq.get("kind") or "pair").strip().lower()
                idx = int(tq.get("index") or 0)
                conf = float(tq.get("confidence") or 0.0)
            except Exception:
                kind = "pair"; idx = 0; conf = 0.0
            if idx > 0 and conf >= conf_min:
                try:
                    try:
                        cap_one = int(os.getenv("JINX_TURNS_MAX_CHARS", "800"))
                    except Exception:
                        cap_one = 800
                    if kind == "user":
                        body = (await _turn_user(idx))
                        if body:
                            if cap_one > 0 and len(body) > cap_one:
                                body = body[:cap_one]
                            turns_block = f"<turns>\n[User:{idx}]\n{body}\n</turns>"
                    elif kind == "jinx":
                        body = (await _turn_jinx(idx))
                        if body:
                            if cap_one > 0 and len(body) > cap_one:
                                body = body[:cap_one]
                            turns_block = f"<turns>\n[Jinx:{idx}]\n{body}\n</turns>"
                    else:
                        turns = await _turns_all()
                        if 0 < idx <= len(turns):
                            u = (turns[idx-1].get("user") or "").strip()
                            a = (turns[idx-1].get("jinx") or "").strip()
                            try:
                                cap_pair = int(os.getenv("JINX_TURNS_PAIR_MAX_CHARS", "1200"))
                            except Exception:
                                cap_pair = 1200
                            tiny = (u + "\n" + a).strip()
                            if cap_pair > 0 and len(tiny) > cap_pair:
                                tiny = tiny[:cap_pair]
                            turns_block = f"<turns>\n[Pair:{idx}]\n{tiny}\n</turns>"
                except Exception:
                    turns_block = ""

        # Optional: memory program — plan+execute ops (memroute/pins/topics/channels). Prefer this over simple selector.
        memsel_block = ""
        prog_blocks: dict[str, str] = {}
        try:
            if _likely_mem(x or ""):
                prog_blocks = await _mem_program(x or "")
        except Exception:
            prog_blocks = {}
        # Merge any blocks returned by program
        if prog_blocks:
            try:
                if prog_blocks.get("memory_selected"):
                    memsel_block = prog_blocks["memory_selected"].strip()
                if prog_blocks.get("pins"):
                    pins_block = prog_blocks["pins"].strip()
                    memsel_block = (memsel_block + "\n\n" + pins_block).strip() if memsel_block else pins_block
            except Exception:
                memsel_block = memsel_block or ""

        # If program produced nothing, fall back to memory reasoner — decide if routed memory or pins should be injected compactly
        ma = None
        if not memsel_block:
            try:
                if _likely_mem(x or ""):
                    ma = await _infer_memsel(x or "")
            except Exception:
                ma = None
        if (not memsel_block) and ma:
            try:
                mconf_min = float(os.getenv("JINX_MEMSEL_CONF_MIN", "0.4"))
            except Exception:
                mconf_min = 0.4
            action = str(ma.get("action") or "").strip().lower()
            params = ma.get("params") or {}
            try:
                mconf = float(ma.get("confidence") or 0.0)
            except Exception:
                mconf = 0.0
            if mconf >= mconf_min:
                try:
                    if action == "memroute":
                        q = str(params.get("query") or "")
                        try:
                            kk = int(params.get("k") or 0)
                        except Exception:
                            kk = 0
                        if kk <= 0:
                            try:
                                kk = int(os.getenv("JINX_MEMSEL_K", "8"))
                            except Exception:
                                kk = 8
                        kk = max(1, min(16, kk))
                        try:
                            pv = int(os.getenv("JINX_MEMSEL_PREVIEW_CHARS", os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
                            pv = max(24, pv)
                        except Exception:
                            pv = 160
                        lines = await _memroute(q, k=kk, preview_chars=pv)
                        body = "\n".join([ln for ln in (lines or [])[:kk] if ln])
                        try:
                            cap = int(os.getenv("JINX_MEMSEL_MAX_CHARS", "1200"))
                        except Exception:
                            cap = 1200
                        if body and cap > 0 and len(body) > cap:
                            body = body[:cap]
                        if body:
                            memsel_block = f"<memory_selected>\n{body}\n</memory_selected>"
                    elif action == "pins":
                        try:
                            pins = _pins_load()
                        except Exception:
                            pins = []
                        if pins:
                            body = "\n".join(pins[:8])
                            memsel_block = f"<pins>\n{body}\n</pins>"
                except Exception:
                    memsel_block = ""
        elif _likely_mem(x or ""):
            # Fallback path: inject a minimal memroute block using the whole question as query
            try:
                await log_debug("JINX_MEMSEL", "fallback_memroute")
                try:
                    fb_k = int(os.getenv("JINX_MEMSEL_FALLBACK_K", "4"))
                except Exception:
                    fb_k = 4
                fb_k = max(1, min(8, fb_k))
                try:
                    pv = int(os.getenv("JINX_MEMSEL_PREVIEW_CHARS", os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
                    pv = max(24, pv)
                except Exception:
                    pv = 160
                q = (x or "").strip()[:240]
                lines = await _memroute(q, k=fb_k, preview_chars=pv)
                body = "\n".join([ln for ln in (lines or [])[:fb_k] if ln])
                if body:
                    try:
                        cap = int(os.getenv("JINX_MEMSEL_MAX_CHARS", "900"))
                    except Exception:
                        cap = 900
                    if len(body) > cap:
                        body = body[:cap]
                    memsel_block = f"<memory_selected>\n{body}\n</memory_selected>"
            except Exception:
                pass

        # Do NOT send <embeddings_memory> to API by default; keep only base/project/planner/continuity (+ optional <turns>/<memory_selected>)
        ctx = "\n".join([c for c in [base_ctx, proj_ctx, plan_ctx, cont_block, turns_block, memsel_block] if c])

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
        # 2) <memory> from file-based view (active.md or active.compact.md). Default ON.
        try:
            is_followup = _is_short(x or "")
        except Exception:
            is_followup = False
        try:
            mem_text = ""
            if str(os.getenv("JINX_MEMORY_BLOCK_SEND", "1")).lower() not in ("", "0", "false", "off", "no"):
                mem_text = await _build_api_mem(is_followup, topic_shifted)
        except Exception:
            mem_text = ""
        # 2.5) <evergreen> persistent durable facts
        # Default: do NOT include evergreen content in the LLM payload.
        # If explicitly enabled via JINX_EVERGREEN_SEND=1, include a compact selection.
        evergreen_text = ""
        try:
            send_evg = str(os.getenv("JINX_EVERGREEN_SEND", "0")).lower() not in ("", "0", "false", "off", "no")
            if send_evg:
                q_for_evg = _q if '_q' in locals() else (x or "")
                evergreen_text = await _select_evg(q_for_evg, anchors=anchors if 'anchors' in locals() else None)
        except Exception:
            evergreen_text = ""
        # Continuity: optionally gate evergreen (when sending) by topic shift on short follow-ups
        if evergreen_text:
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
        # Streaming fast-path (env-gated): early-run on first complete code block
        executed_early: bool = False
        printed_tail_early: bool = False
        stream_on = str(os.getenv("JINX_LLM_STREAM_FASTPATH", "1")).lower() not in ("", "0", "false", "off", "no")

        # Early execution callback (receives code body and code_id)
        async def _early_exec(body: str, cid: str) -> None:
            nonlocal executed_early
            nonlocal printed_tail_early
            if executed_early:
                return
            # Heuristic guard: skip early run when the first complete block is not code-like
            # (e.g., model emitted <python_question_...> or prose instead of executable code)
            try:
                if not _is_code_like(body or ""):
                    return
            except Exception:
                # Fail-closed: if heuristic unavailable, do not early-execute
                return
            minimal = f"<python_{cid}>\n{body}\n</python_{cid}>"
            async def _early_err(e: Optional[str]) -> None:
                if not e:
                    return
                try:
                    pretty_echo(minimal)
                    await show_sandbox_tail()
                except Exception:
                    pass
                # Attach the executed code to the error payload so recovery sees the code to fix
                payload = _attach_error_code(e or "", None, cid, code_body=body)
                try:
                    await corrupt_report(payload)
                except Exception:
                    pass
            try:
                ok = await run_blocks(minimal, cid, _early_err)
                if ok:
                    executed_early = True
                    await show_sandbox_tail()
                    printed_tail_early = True
                    try:
                        await embed_text(minimal.strip(), source="dialogue", kind="agent")
                    except Exception:
                        pass
            except Exception:
                pass

        if stream_on:
            out, code_id = await _spark_llm_stream(chains, prompt_override=prompt_override, on_first_block=_early_exec)
        else:
            out, code_id = await _spark_llm(chains, prompt_override=prompt_override)
        # Normalize model output to ensure exactly one <python_{code_id}> block and proper fences
        try:
            out = normalize_output_blocks(out, code_id)
        except Exception:
            pass
        # Always show the model output box for the current turn (once)
        printed_out_box: bool = False
        try:
            pretty_echo(out)
            printed_out_box = True
        except Exception:
            pass

        # Ensure that on any execution error we also show the raw model output
        async def on_exec_error(err_msg: Optional[str]) -> None:
            # Sandbox callback sends None on success — ignore to avoid duplicate log prints
            if not err_msg:
                return
            # Avoid re-printing the same model box; it's already shown above
            await show_sandbox_tail()
            # Attach the executed code to the error payload so recovery sees the code to fix
            payload = _attach_error_code(err_msg or "", out, code_id)
            await corrupt_report(payload)

        # If early executed successfully, treat as executed to prevent duplicate run/print
        executed = True if executed_early else await run_blocks(out, code_id, on_exec_error)
        if not executed:
            await bomb_log(f"No executable <python_{code_id}> block found in model output; displaying raw output.")
            # Already printed above
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
            # Append turn to file-based memory (best-effort)
            try:
                await _append_turn((x or ""), (out or ""))
            except Exception:
                pass
        else:
            # After successful execution, also surface the latest sandbox log context (avoid duplicate if already printed early)
            if not printed_tail_early:
                await show_sandbox_tail()
            # Also embed the agent output for retrieval (source: dialogue)
            try:
                await embed_text(out.strip(), source="dialogue", kind="agent")
            except Exception:
                pass
            # Append turn to file-based memory (best-effort)
            try:
                await _append_turn((x or ""), (out or ""))
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
