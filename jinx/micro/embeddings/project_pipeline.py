from __future__ import annotations

import asyncio
import json
import os
import hashlib
from typing import Any, Dict, List, Tuple
import contextlib

from .project_paths import (
    ensure_project_dirs,
    PROJECT_FILES_DIR,
    PROJECT_INDEX_DIR,
    safe_rel_path,
)
from .project_chunk_char import chunk_text_char
from .project_chunk_token import chunk_text_token
from .project_chunk_types import Chunk
from .project_terms import extract_terms
from .project_io import write_json_atomic
from .util import now_ts
from .embed_cache import embed_texts_cached, embed_text_cached
from .project_chunk_semantic import chunk_text_semantic
from .digest import make_digest
from jinx.async_utils.fs import read_text_abs_thread
from .fingerprint import simhash
from .api_lens import find_endpoints as _api_eps, find_models as _api_models


MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNKER_KIND = os.getenv("EMBED_PROJECT_CHUNKER", "char").strip().lower()


async def _embed_text(text: str) -> List[float]:
    try:
        return await embed_text_cached(text, model=MODEL)
    except Exception:
        return []


async def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-create embeddings via shared cache layer (TTL, coalescing, limits).

    Preserves order; returns empty vectors on failure.
    """
    try:
        return await embed_texts_cached(texts, model=MODEL)
    except Exception:
        return [[] for _ in texts]

def _file_terms(full_text: str) -> List[str]:
    return extract_terms(full_text)


def _chunk_text(text: str) -> List[Chunk]:
    """Select chunker per env; fallback to char-based if token chunker unavailable."""
    if CHUNKER_KIND == "token":
        toks = chunk_text_token(text)
        if toks:
            return toks
    if CHUNKER_KIND == "semantic":
        sems = chunk_text_semantic(text)
        if sems:
            return sems
    return chunk_text_char(text)


async def embed_file(abs_path: str, rel_path: str, *, file_sha: str, prune_old: bool = True) -> Dict[str, Any]:
    """Build embeddings for a single file into emb/.

    Creates per-chunk JSONs at: emb/files/<safe_rel>/<chunk_sha>.json
    And a file index at:        emb/index/<safe_rel>.json
    """
    ensure_project_dirs()

    safe = safe_rel_path(rel_path)
    file_dir = os.path.join(PROJECT_FILES_DIR, safe)
    os.makedirs(file_dir, exist_ok=True)

    # Read file off the event loop (shared helper with LRU)
    try:
        text = await read_text_abs_thread(abs_path)
    except Exception:
        # If unreadable, write an empty index and return
        index_path = os.path.join(PROJECT_INDEX_DIR, f"{safe}.json")
        data = {
            "file_rel": rel_path,
            "file_sha256": file_sha,
            "updated_ts": now_ts(),
            "total_chunks": 0,
            "chunks": [],
            "file_terms": [],
        }
        try:
            await asyncio.to_thread(write_json_atomic, index_path, data)
        except Exception:
            pass
        return data

    # Chunking and term extraction can be CPU-heavy; offload
    def _chunk_and_terms(t: str) -> Tuple[List[Chunk], List[str]]:
        return _chunk_text(t), _file_terms(t)
    chunk_items, file_terms = await asyncio.to_thread(_chunk_and_terms, text)

    # Prepare unique chunks and batch-embed
    unique_inputs: List[Tuple[int, str, str, int, int]] = []  # (i, text, sha, line_start, line_end)
    seen_chunks: set[str] = set()
    for i, ch_obj in enumerate(chunk_items):
        ch = str(ch_obj.get("text") or "")
        csha = hashlib.sha256(ch.encode("utf-8", errors="ignore")).hexdigest()
        if csha in seen_chunks:
            continue
        seen_chunks.add(csha)
        ls = int(ch_obj.get("line_start") or 0)
        le = int(ch_obj.get("line_end") or 0)
        unique_inputs.append((i, ch, csha, ls, le))

    batch_texts = [t[1] for t in unique_inputs]
    # Parallelize embeddings and digest generation
    # Do NOT wrap asyncio.gather() into create_task() — create_task expects a coroutine
    vec_fut = asyncio.create_task(_embed_texts(batch_texts))
    dig_fut = asyncio.gather(*(make_digest(t) for t in batch_texts))
    try:
        batch_vecs, batch_digests = await asyncio.gather(vec_fut, dig_fut)
    except asyncio.CancelledError:
        # Ensure child task is awaited to avoid 'Task exception was never retrieved'
        with contextlib.suppress(Exception):
            await vec_fut
        raise

    results: List[Tuple[str, Dict[str, Any]]] = []  # (chunk_sha, payload)
    for idx, (i, ch, csha, ls, le) in enumerate(unique_inputs):
        vec = batch_vecs[idx] if idx < len(batch_vecs) else []
        digest_i = batch_digests[idx] if idx < len(batch_digests) else ""
        fp_i = simhash(ch)
        # Lightweight API metadata (optional)
        try:
            eps = [f"{m} {p} -> {h}()" for (m, p, h) in (_api_eps(ch) or [])][:8]
        except Exception:
            eps = []
        try:
            mdl = (_api_models(ch) or [])[:12]
        except Exception:
            mdl = []
        meta: Dict[str, Any] = {
            "ts": now_ts(),
            "model": MODEL,
            "file_rel": rel_path,
            "chunk_index": i,
            "chunks_total": len(unique_inputs),
            "content_sha256": csha,
            "file_sha256": file_sha,
            "terms": extract_terms(ch),
            "text_preview": ch[:256],
            "dims": len(vec) if vec is not None else 0,
            "line_start": ls,
            "line_end": le,
            "digest": digest_i,
            "fingerprint": fp_i,
            # Optional API hints to help downstream retrieval/reranking
            "api_endpoints": eps,
            "api_models": mdl,
        }
        payload = {"meta": meta, "embedding": vec}
        results.append((csha, payload))

    # Update chunks_total in metadata to reflect unique, persisted chunks
    # chunks_total already set to number of unique inputs; ensure consistent
    total_unique = len(unique_inputs)
    for _, payload in results:
        payload.get("meta", {})["chunks_total"] = total_unique

    # Write new chunk files (atomic replace) off the event loop
    write_tasks: List[asyncio.Task] = []
    for csha, payload in results:
        p = os.path.join(file_dir, f"{csha}.json")
        write_tasks.append(asyncio.create_task(asyncio.to_thread(write_json_atomic, p, payload)))
    if write_tasks:
        await asyncio.gather(*write_tasks, return_exceptions=True)

    # Optionally prune old chunk files so we only keep current set
    if prune_old:
        keep = {f"{csha}.json" for csha, _ in results}
        def _prune(dir_path: str, keep_names: set[str]) -> None:
            try:
                for fn in os.listdir(dir_path):
                    if fn.endswith(".json") and fn not in keep_names:
                        try:
                            os.remove(os.path.join(dir_path, fn))
                        except Exception:
                            pass
            except FileNotFoundError:
                return
            except Exception:
                return
        await asyncio.to_thread(_prune, file_dir, keep)

    # Write/overwrite index file
    index_path = os.path.join(PROJECT_INDEX_DIR, f"{safe}.json")
    index_obj = {
        "file_rel": rel_path,
        "file_sha256": file_sha,
        "updated_ts": now_ts(),
        "total_chunks": len(results),
        "chunks": [
            {
                "sha": csha,
                "index": payload.get("meta", {}).get("chunk_index", i),
                "path": os.path.relpath(os.path.join(file_dir, f"{csha}.json"), start=PROJECT_FILES_DIR),
                "terms": payload.get("meta", {}).get("terms", []),
                "text_preview": payload.get("meta", {}).get("text_preview", ""),
                "line_start": payload.get("meta", {}).get("line_start", 0),
                "line_end": payload.get("meta", {}).get("line_end", 0),
            }
            for i, (csha, payload) in enumerate(results)
        ],
        "file_terms": file_terms,
    }

    try:
        await asyncio.to_thread(write_json_atomic, index_path, index_obj)
    except Exception:
        pass

    return index_obj
