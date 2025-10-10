from __future__ import annotations

import asyncio
import os
from typing import Any

from jinx.logging_service import bomb_log
from jinx.micro.rag.file_search import build_file_search_tools
from jinx.net import get_openai_client
from .llm_cache import call_openai_cached, call_openai_multi_validated
from jinx.micro.text.heuristics import is_code_like as _is_code_like
import asyncio as _asyncio
import queue as _queue


async def call_openai(instructions: str, model: str, input_text: str) -> str:
    """Call OpenAI Responses API and return output text.

    Uses to_thread to run the sync SDK call and relies on the shared retry helper
    at the caller site to provide resiliency.
    """
    try:
        # Auto-adjustment: if no API key is configured, return a graceful stub response
        if not (os.getenv("OPENAI_API_KEY") or ""):
            await bomb_log("OPENAI_API_KEY missing; LLM disabled — returning stub output")
            return (
                "<llm_disabled>\n"
                "No OpenAI API key configured. Set OPENAI_API_KEY in .env to enable model calls.\n"
                "</llm_disabled>"
            )
        # Heuristic: enable File Search tools only for code-like queries unless gated off
        try:
            fs_gate_on = (os.getenv("JINX_FILESEARCH_GATE", "1").strip().lower() not in ("", "0", "false", "off", "no"))
        except Exception:
            fs_gate_on = True
        extra_kwargs: dict[str, Any]
        if fs_gate_on and not _is_code_like(input_text or ""):
            extra_kwargs = {}
        else:
            extra_kwargs = build_file_search_tools()
        # Legacy single-sample path
        return await call_openai_cached(
            instructions=instructions,
            model=model,
            input_text=input_text,
            extra_kwargs=extra_kwargs,
        )
    except Exception as e:
        await bomb_log(f"ERROR cortex exploded: {e}")
        raise


async def call_openai_validated(instructions: str, model: str, input_text: str, *, code_id: str) -> str:
    """Preferred LLM path: multi-sample race with strict validation and TTL cache.

    Enabled by default via env (JINX_LLM_MULTI_ENABLE=1). Falls back to single-sample
    cached call when disabled.
    """
    try:
        multi_on = (os.getenv("JINX_LLM_MULTI_ENABLE", "1").strip().lower() not in ("", "0", "false", "off", "no"))
    except Exception:
        multi_on = True
    # Heuristic: enable File Search tools only for code-like queries unless gated off
    try:
        fs_gate_on = (os.getenv("JINX_FILESEARCH_GATE", "1").strip().lower() not in ("", "0", "false", "off", "no"))
    except Exception:
        fs_gate_on = True
    if fs_gate_on and not _is_code_like(input_text or ""):
        extra_kwargs: dict[str, Any] = {}
    else:
        extra_kwargs = build_file_search_tools()
    if multi_on:
        return await call_openai_multi_validated(
            instructions=instructions,
            model=model,
            input_text=input_text,
            code_id=code_id,
            base_extra_kwargs=extra_kwargs,
        )
    # Fallback
    return await call_openai_cached(
        instructions=instructions,
        model=model,
        input_text=input_text,
        extra_kwargs=extra_kwargs,
    )


async def call_openai_stream_first_block(
    instructions: str,
    model: str,
    input_text: str,
    *,
    code_id: str,
    on_first_block: callable | None = None,
) -> str:
    """Stream Responses API, fire early when first complete <python_{code_id}> block appears.

    Fallback to validated non-stream call on any streaming error.
    """
    # File Search gating
    try:
        fs_gate_on = (os.getenv("JINX_FILESEARCH_GATE", "1").strip().lower() not in ("", "0", "false", "off", "no"))
    except Exception:
        fs_gate_on = True
    if fs_gate_on and not _is_code_like(input_text or ""):
        extra_kwargs: dict[str, Any] = {}
    else:
        extra_kwargs = build_file_search_tools()

    ltag = f"<python_{code_id}>"
    rtag = f"</python_{code_id}>"
    buf: list[str] = []
    fired = False

    def _worker(q: _queue.Queue[str]) -> None:
        try:
            client = get_openai_client()
            # Prefer streaming API if available in SDK
            stream_fn = getattr(getattr(client, "responses", client), "stream", None)
            if stream_fn is None:
                raise RuntimeError("responses.stream_not_supported")
            with client.responses.stream(
                instructions=instructions,
                model=model,
                input=input_text,
                **{k: v for k, v in (extra_kwargs or {}).items() if not str(k).startswith("__")},
            ) as stream:
                for event in stream:
                    try:
                        typ = getattr(event, "type", "") or ""
                    except Exception:
                        typ = ""
                    piece = ""
                    # Common event types in Responses streaming
                    if typ.endswith(".delta"):
                        piece = getattr(event, "delta", "") or ""
                    elif typ.endswith("output_text"):
                        piece = getattr(event, "output_text", "") or getattr(event, "text", "") or ""
                    else:
                        piece = getattr(event, "delta", "") or getattr(event, "text", "") or ""
                    if piece:
                        q.put(piece)
            q.put("__DONE__")
        except Exception as e:
            q.put(f"__ERROR__:{e}")

    q: _queue.Queue[str] = _queue.Queue()
    # Run the streaming worker in a thread
    worker_task = _asyncio.create_task(_asyncio.to_thread(_worker, q))

    async def _get_next() -> str:
        return await _asyncio.to_thread(q.get)

    try:
        while True:
            chunk = await _get_next()
            if chunk == "__DONE__":
                break
            if isinstance(chunk, str) and chunk.startswith("__ERROR__:"):
                raise RuntimeError(chunk)
            if chunk:
                buf.append(chunk)
                if (not fired) and (ltag in chunk or buf):
                    text = "".join(buf)
                    li = text.find(ltag)
                    if li != -1:
                        ri = text.find(rtag, li + len(ltag))
                        if ri != -1:
                            body = text[li + len(ltag): ri]
                            if (body or "").strip():
                                fired = True
                                if on_first_block:
                                    try:
                                        _asyncio.create_task(on_first_block(body, code_id))
                                    except Exception:
                                        pass
        # Join worker
        try:
            await worker_task
        except Exception:
            pass
        return "".join(buf)
    except Exception:
        # Fallback to validated path (single outbound call if streaming never started)
        return await call_openai_validated(instructions, model, input_text, code_id=code_id)
