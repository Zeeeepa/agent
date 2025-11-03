from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST
from jinx.micro.embeddings.search_cache import search_project_cached


class AutoVerifyProgram(MicroProgram):
    """Embedding-based verifier program.

    Handles TASK_REQUEST "verify.embedding" with payload:
    { id, name, args: [], kwargs: { goal: str, files?: List[str], diff?: str, topk?: int } }

    Produces a score [0..1] based on whether changed files/snippets are retrieved by embeddings
    for the user goal. Publishes human-readable reason and saves exports for prompt macros.
    """

    def __init__(self) -> None:
        super().__init__(name="AutoVerifyProgram")
        self.exports: Dict[str, str] = {}

    def get_export(self, key: str) -> str:
        k = str(key or "").strip().lower()
        val = self.exports.get(k)
        if not val:
            return ""
        try:
            cap = max(256, int(os.getenv("JINX_VERIFY_EXPORT_MAXCHARS", "2000")))
        except Exception:
            cap = 2000
        return val if len(val) <= cap else (val[:cap] + "\n...<truncated>")

    async def run(self) -> None:
        from jinx.micro.runtime.api import on as _on
        await _on(TASK_REQUEST, self._on_task)
        await self.log("verifier online")
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: dict) -> None:
        try:
            name = str(payload.get("name") or "")
            tid = str(payload.get("id") or "")
            if name != "verify.embedding" or not tid:
                return
            kw = payload.get("kwargs") or {}
            goal = str(kw.get("goal") or "").strip()
            files: List[str] = [str(x) for x in (kw.get("files") or [])]
            diff = str(kw.get("diff") or "")
            try:
                topk = int(kw.get("topk")) if kw.get("topk") is not None else int(os.getenv("JINX_VERIFY_TOPK", "6"))
            except Exception:
                topk = 6
            await self._handle_verify_embedding(tid, goal, files, diff, topk)
        except Exception:
            pass

    async def _handle_verify_embedding(self, tid: str, goal: str, files: List[str], diff: str, topk: int) -> None:
        try:
            from jinx.micro.runtime.api import report_progress as _report_progress, report_result as _report_result
            if not goal:
                await _report_result(tid, False, error="goal required")
                return
            await _report_progress(tid, 10.0, "searching project")
            hits = await search_project_cached(goal, k=max(1, topk), max_time_ms=int(os.getenv("JINX_VERIFY_MS", "400")))
            # simple scoring: +0.5 if any file matches, +0.3 if multi matches, +0.2 if diff mentions headers
            score = 0.0
            matched_files: List[str] = []
            if hits:
                files_norm = {str(f).replace("\\", "/").strip() for f in files or []}
                for h in hits:
                    f = str(h.get("file") or "").replace("\\", "/")
                    if f and f in files_norm:
                        matched_files.append(f)
                if matched_files:
                    score += 0.5
                    # if multiple files matched, increase
                    if len(matched_files) >= 2:
                        score += 0.3
            # diff header heuristic
            try:
                has_header_ref = any((str(h.get("header") or "") in diff) for h in (hits or []))
            except Exception:
                has_header_ref = False
            if has_header_ref:
                score += 0.2
            # mild clamp
            score = max(0.0, min(1.0, score))
            try:
                pass_thr = float(os.getenv("JINX_VERIFY_PASS", "0.6"))
            except Exception:
                pass_thr = 0.6
            ok = bool(score >= pass_thr)
            reason = f"score={score:.2f} pass_thr={pass_thr:.2f}; matched_files={matched_files or []}"
            self.exports["last_verify_score"] = f"{score:.2f}"
            self.exports["last_verify_reason"] = reason
            if matched_files:
                self.exports["last_verify_files"] = ", ".join(matched_files)
            await _report_result(tid, ok, {"score": score, "matched_files": matched_files, "topk": topk}, None if ok else "below threshold")
        except Exception as e:
            from jinx.micro.runtime.api import report_result as _report_result
            await _report_result(tid, False, error=f"verify failed: {e}")


# Helpers
async def spawn_verifier() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(AutoVerifyProgram())


async def ensure_verifier_running() -> Optional[str]:
    global _VERIFIER_PID, _VERIFIER_STARTED
    try:
        if _VERIFIER_STARTED and _VERIFIER_PID:
            return _VERIFIER_PID
        # We cannot inspect program names reliably; cache pid locally
        pid = await spawn_verifier()
        _VERIFIER_PID = pid
        _VERIFIER_STARTED = True
        return pid
    except Exception:
        return None


# globals for verifier lifecycle
_VERIFIER_PID: Optional[str] = None
_VERIFIER_STARTED: bool = False

async def submit_verify_embedding(goal: str, files: Optional[List[str]] = None, diff: str = "", *, topk: Optional[int] = None) -> str:
    from jinx.micro.runtime.api import submit_task as _submit_task
    return await _submit_task(
        "verify.embedding",
        goal=str(goal or ""),
        files=list(files or []),
        diff=str(diff or ""),
        topk=int(topk) if (topk is not None) else None,
    )
