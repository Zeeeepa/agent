from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST


def _lang_for(path: str) -> str:
    p = (path or "").lower()
    if p.endswith(".py"): return "py"
    if p.endswith(".ts"): return "ts"
    if p.endswith(".tsx"): return "tsx"
    if p.endswith(".js"): return "js"
    if p.endswith(".jsx"): return "jsx"
    if p.endswith(".sh"): return "sh"
    if p.endswith(".go"): return "go"
    return ""


def _read_snippet(abs_path: str, center_line: Optional[int] = None, radius: int = 20) -> str:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        if not lines:
            return ""
        if center_line is None:
            a = 0
            b = min(len(lines), 1 + 2 * radius)
        else:
            a = max(1, int(center_line) - radius)
            b = min(len(lines), int(center_line) + radius)
        return "\n".join(lines[a - 1 : b])
    except Exception:
        return ""


async def _ast_sanity(abs_path: str) -> Tuple[bool, str]:
    if not abs_path.lower().endswith(".py"):
        return True, ""
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        return False, f"read_error: {e}"
    try:
        import ast
        ast.parse(code)
    except Exception as e:
        return False, f"ast_error: {e}"
    # Optional libcst if present
    try:
        import libcst as cst  # type: ignore
        cst.parse_module(code)
    except Exception:
        # treat libcst failure as soft warn; do not block
        pass
    # py_compile smoke
    try:
        import py_compile
        py_compile.compile(abs_path, doraise=True)
    except Exception as e:
        return False, f"compile_error: {e}"
    return True, ""


class QualityScanner(MicroProgram):
    """RT-aware code quality scanner. Receives TASK_REQUEST 'quality.scan'.

    - Runs small static checks (AST/libcst/py_compile) under budgets.
    - Optionally calls OpenAI audit prompt (code_audit) with strict JSON.
    - Publishes quality.issue/quality.summary via plugin bus.
    - Best-effort, never crashes; respects risk policies.
    """

    def __init__(self) -> None:
        super().__init__(name="QualityScanner")
        try:
            self._budget_ms = int(os.getenv("JINX_QUALITY_BUDGET_MS", "700"))
        except Exception:
            self._budget_ms = 700
        try:
            self._cache_ttl = float(os.getenv("JINX_QUALITY_CACHE_TTL_SEC", "900"))
        except Exception:
            self._cache_ttl = 900.0
        # Simple in-memory cache: key=(abs_path, mtime)-> {ts, issues_static}
        self._cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

    async def run(self) -> None:
        from jinx.micro.runtime.api import on as _on
        await _on(TASK_REQUEST, self._on_task)
        await self.log("quality-scanner online")
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: Dict[str, Any]) -> None:
        name = str(payload.get("name") or "")
        if name != "quality.scan":
            return
        tid = str(payload.get("id") or "")
        if not tid:
            return
        kw = payload.get("kwargs") or {}
        files = kw.get("files") or []
        try:
            max_ms = int(kw.get("max_ms") or self._budget_ms)
        except Exception:
            max_ms = self._budget_ms
        await self._scan(files, budget_ms=max_ms)

    async def _scan(self, files: List[str], *, budget_ms: int) -> None:
        # Publish helper (non-blocking)
        try:
            from jinx.micro.runtime.plugins import publish_event as _pub
        except Exception:
            def _pub(_t: str, _p: Any) -> None:  # type: ignore
                return
        # Resolve targets
        targets: List[Tuple[str, Optional[int]]] = []
        cwd = os.getcwd()
        for f in files or []:
            p = f
            if not os.path.isabs(p):
                p = os.path.join(cwd, p)
            line = None
            m = re.match(r"^(.*?):(\d+)$", f)
            if m:
                p = m.group(1)
                if not os.path.isabs(p):
                    p = os.path.join(cwd, p)
                try:
                    line = int(m.group(2))
                except Exception:
                    line = None
            targets.append((p, line))
        # If none provided, skip silently
        if not targets:
            _pub("quality.summary", {"files_scanned": 0, "elapsed_ms": 0, "counts_by_severity": {}})
            return
        # Risk Policy
        try:
            from jinx.micro.runtime.risk_policies import deny_patterns as _deny
            deny = _deny() or []
        except Exception:
            deny = []
        def _allowed(rel: str) -> bool:
            if not deny:
                return True
            for pat in deny:
                try:
                    import fnmatch
                    if fnmatch.fnmatch(rel, pat):
                        return False
                except Exception:
                    pass
            return True
        # Static checks (fast path)
        issues: List[Dict[str, Any]] = []
        start = asyncio.get_running_loop().time()
        for (ap, line) in targets:
            rel = os.path.relpath(ap, cwd)
            if not _allowed(rel):
                continue
            # Cache lookup by mtime
            try:
                st = os.stat(ap)
                key = (ap, int(st.st_mtime))
            except Exception:
                key = (ap, 0)
            cached = self._cache.get(key)
            now = asyncio.get_running_loop().time()
            cached_ok = bool(cached and (now - float(cached.get("ts") or 0.0) < self._cache_ttl))
            if cached_ok:
                for it in cached.get("issues_static", []):
                    # rewrite path to current rel
                    dit = dict(it)
                    dit["file"] = rel
                    issues.append(dit)
            else:
                ok, msg = await _ast_sanity(ap)
                i_static: List[Dict[str, Any]] = []
                if not ok:
                    i_static.append({
                        "file": rel,
                        "line": None,
                        "kind": "static",
                        "severity": "error",
                        "message": msg,
                        "tool": "ast_sanity",
                        "fingerprint": f"ast:{rel}",
                    })
                # Secrets scan
                for sec in await self._secrets_scan(ap, rel):
                    i_static.append(sec)
                # Optional type check (mypy) under tight budget
                for typ in await self._typecheck_scan(ap, rel, budget_ms=200):
                    i_static.append(typ)
                # Persist cache entry
                self._cache[key] = {"ts": now, "issues_static": i_static}
                issues.extend(i_static)
        # LLM audit (budgeted)
        llm_issues: List[Dict[str, Any]] = []
        try:
            # Prepare files payload (limit size)
            flist: List[Dict[str, Any]] = []
            cap = max(1, min(len(targets), 6))
            for (ap, line) in targets[:cap]:
                rel = os.path.relpath(ap, cwd)
                if not _allowed(rel):
                    continue
                lang = _lang_for(rel)
                snippet = _read_snippet(ap, center_line=line)
                if snippet:
                    # Truncate to ~1200 chars
                    snippet = snippet[:1200]
                flist.append({"path": rel, "lang": lang, "snippet": snippet})
            # Compose prompt
            files_json = json.dumps(flist, ensure_ascii=True)
            policy_txt = "DENY: " + (", ".join(deny) if deny else "(none)")
            from jinx.prompts import render_prompt as _render
            instructions = _render("code_audit", files_json=files_json, policy=policy_txt, budget_ms=budget_ms)
            # Admission and call
            try:
                from jinx.rt.admission import guard as _guard
            except Exception:
                _guard = None  # type: ignore
            from jinx.micro.llm.service import spark_openai as _spark
            if _guard is not None:
                async with _guard("llm", timeout_ms=min(300, budget_ms)) as admitted:
                    if admitted:
                        out, _ = await asyncio.wait_for(_spark(instructions), timeout=max(0.1, budget_ms / 1000.0))
                    else:
                        out = ""
            else:
                out, _ = await asyncio.wait_for(_spark(instructions), timeout=max(0.1, budget_ms / 1000.0))
            if out:
                # Extract JSON
                try:
                    m = re.search(r"\{[\s\S]*\}", out)
                    s = m.group(0) if m else out.strip()
                    data = json.loads(s)
                except Exception:
                    data = {}
                for it in data.get("issues", []) if isinstance(data, dict) else []:
                    if not isinstance(it, dict):
                        continue
                    file_rel = str(it.get("file") or "").strip()
                    if not file_rel or not _allowed(file_rel):
                        continue
                    llm_issues.append({
                        "file": file_rel,
                        "line": it.get("line"),
                        "symbol": it.get("symbol"),
                        "kind": str(it.get("kind") or "audit"),
                        "severity": str(it.get("severity") or "warn"),
                        "message": str(it.get("message") or ""),
                        "proposed_strategy": str(it.get("proposed_strategy") or ""),
                        "tool": "code_audit",
                        "fingerprint": f"audit:{file_rel}:{it.get('kind')}:{it.get('line')}",
                        "patch": it.get("patch"),
                    })
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        # Publish issues
        for iss in issues + llm_issues:
            try:
                _pub("quality.issue", iss)
            except Exception:
                pass
        # Summary
        elapsed_ms = int((asyncio.get_running_loop().time() - start) * 1000)
        counts: Dict[str, int] = {"info": 0, "warn": 0, "error": 0, "critical": 0}
        for it in issues + llm_issues:
            sev = str(it.get("severity") or "warn").lower()
            if sev not in counts:
                sev = "warn"
            counts[sev] += 1
        try:
            _pub("quality.summary", {"files_scanned": len(targets), "elapsed_ms": elapsed_ms, "counts_by_severity": counts})
        except Exception:
            pass

    async def _secrets_scan(self, abs_path: str, rel: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not abs_path.lower().endswith((".py", ".env", ".txt", ".cfg", ".ini", ".json", ".yaml", ".yml")):
            return out
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return out
        findings: List[Tuple[int, str]] = []
        # Simple regexes (API keys, tokens)
        patterns = [
            r"aws_secret_access_key\s*[:=]\s*[A-Za-z0-9/+=]{20,}",
            r"api_key\s*[:=]\s*[A-Za-z0-9_-]{20,}",
            r"token\s*[:=]\s*[A-Za-z0-9_-]{24,}",
            r"-----BEGIN (?:RSA|DSA|EC) PRIVATE KEY-----",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                findings.append((text.count("\n", 0, m.start()) + 1, f"pattern:{pat[:20]}"))
        # Entropy check for suspicious long strings
        for m in re.finditer(r"[A-Za-z0-9/_+=-]{32,}", text):
            s = m.group(0)
            # Quick entropy estimate
            try:
                import math
                p = {}
                for ch in s:
                    p[ch] = p.get(ch, 0) + 1
                n = len(s)
                h = -sum((c / n) * math.log2(c / n) for c in p.values())
                if h >= 3.5:  # heuristic
                    findings.append((text.count("\n", 0, m.start()) + 1, "entropy"))
            except Exception:
                pass
        for (ln, kind) in findings[:6]:
            out.append({
                "file": rel,
                "line": ln,
                "kind": "secrets",
                "severity": "error" if kind.startswith("pattern") else "warn",
                "message": f"potential secret ({kind})",
                "tool": "secrets_scan",
                "fingerprint": f"secret:{rel}:{ln}:{kind}",
            })
        return out

    async def _typecheck_scan(self, abs_path: str, rel: str, *, budget_ms: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not abs_path.lower().endswith(".py"):
            return out
        # Optional mypy run (in-process API) with timeout; swallow errors
        try:
            import mypy.api as mypy_api  # type: ignore
        except Exception:
            return out
        async def _run() -> Tuple[str, str, int]:
            try:
                return await asyncio.to_thread(mypy_api.run, [abs_path])  # (stdout, stderr, exit_status)
            except Exception:
                return ("", "", 0)
        try:
            stdout, stderr, code = await asyncio.wait_for(_run(), timeout=max(0.1, budget_ms / 1000.0))
            if code != 0 and stdout:
                # Parse typical lines: path:line: column: error: message  [code]
                for ln in stdout.splitlines()[:8]:
                    m = re.match(r"^(.*?):(\d+):\s*(?:\d+:)?\s*error:\s*(.*)$", ln)
                    if not m:
                        continue
                    try:
                        line_no = int(m.group(2))
                    except Exception:
                        line_no = None
                    msg = m.group(3).strip()
                    out.append({
                        "file": rel,
                        "line": line_no,
                        "kind": "typecheck",
                        "severity": "warn",
                        "message": msg,
                        "tool": "mypy",
                        "fingerprint": f"mypy:{rel}:{line_no}:{hash(msg)%10000}",
                    })
        except Exception:
            pass
        return out


async def spawn_quality_scanner() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(QualityScanner())
