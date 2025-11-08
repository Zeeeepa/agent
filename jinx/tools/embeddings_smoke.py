from __future__ import annotations

import asyncio
import json
import sys


async def main() -> int:
    ok = True
    try:
        from jinx.micro.embeddings.pipeline import embed_text
        obj = await embed_text("hello embeddings", source="smoke", kind="line")
        print("embed_text: ", bool(obj and isinstance(obj, dict) and obj.get("embedding")))
    except Exception as e:
        print("embed_text: EXC", e)
        ok = False
    try:
        from jinx.micro.embeddings.vector_stage_semantic import semantic_search
        hits = await semantic_search("fastapi router", k=3, max_time_ms=200)
        print("semantic_search: ", len(hits))
    except Exception as e:
        print("semantic_search: EXC", e)
        ok = False
    try:
        from jinx.micro.embeddings.retrieval_core import retrieve_project_top_k
        r = await retrieve_project_top_k("create endpoint", k=3, max_time_ms=250)
        print("retrieval_core: ", len(r))
    except Exception as e:
        print("retrieval_core: EXC", e)
        ok = False
    try:
        from jinx.micro.embeddings.unified_context import build_unified_context_for
        ctx = await build_unified_context_for("implement POST /users", max_time_ms=350)
        print("unified_context: ", len(ctx))
    except Exception as e:
        print("unified_context: EXC", e)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
