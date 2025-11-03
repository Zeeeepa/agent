"""Advanced local memory builder with optimization and caching."""

from __future__ import annotations

import os
import re
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from functools import lru_cache
from dataclasses import dataclass

from jinx.config import ALL_TAGS
from jinx.micro.embeddings.project_identifiers import extract_identifiers
from jinx.micro.memory.storage import ensure_nl
from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT
from jinx.micro.memory.facts_store import load_facts as _facts_load, save_facts as _facts_save
from jinx.micro.memory.pin_store import load_pins as _load_pins
from jinx.micro.text.heuristics import (
    is_code_like_line as _is_code_like_line,
    extract_preference_fragments as _extract_pref_frags,
    extract_decision_fragments as _extract_decision_frags,
)
from jinx.micro.memory.usage_store import weight_for as _usage_weight, last_used_ms as _last_used_ms, count_for as _usage_count


# -------- Paths --------

def _memory_dir() -> str:
    try:
        sub = os.getenv("JINX_MEMORY_DIR", os.path.join(".jinx", "memory"))
    except Exception:
        sub = os.path.join(".jinx", "memory")
    root = PROJECT_ROOT or os.getcwd()
    return os.path.join(root, sub)

_MEM_DIR = _memory_dir()


# -------- Utilities --------

# Cached regex compilation for performance
@lru_cache(maxsize=1)
def _get_tool_block_regex() -> re.Pattern:
    """Build and cache tool block regex."""
    tag_alt = "|".join(sorted(ALL_TAGS))
    return re.compile(fr"<(?:{tag_alt})_[^>]+>.*?</(?:{tag_alt})_[^>]+>", re.DOTALL)

# Pre-compiled patterns
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\r\n]+|(?:\.|\.{1,2})?/(?:[^\s/]+/)*[^\s/]+\.[A-Za-z0-9]{1,6})")
_SET_RE = re.compile(r"\b([A-Z_]{2,})\s*=\s*([^\s,;]+)")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+\b")
_TOKEN_RE = re.compile(r"\b[A-Za-z0-9+/=_-]{24,}\b")
_ROLE_RE = re.compile(r"^\s*(User|Jinx|Error|State|Note):\s*(.*)$", re.IGNORECASE)
_TRAILING_ROLE_RE = re.compile(r"\s+(?:User|Jinx):\s*$")
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _strip_tool_blocks(text: str) -> Tuple[str, List[str]]:
    """Strip tool blocks and return cleaned text with extracted blocks.
    
{{ ... }}
    Optimizations:
    - Cached regex
    - Single pass through text
    - Efficient string operations
    """
    if not text:
        return "", []
    
    blocks: List[str] = []
    tool_block_re = _get_tool_block_regex()
    
    def _collect(m: re.Match) -> str:
        blk = m.group(0)
        if blk and blk.strip():
            blocks.append(blk.strip())
        return ""
    
    # Remove tool blocks
    cleaned = tool_block_re.sub(_collect, text)
    
    # Collapse excessive blank lines (use cached regex)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned).strip()
    
    return cleaned, blocks


def _dedupe_consecutive(lines: List[str]) -> List[str]:
    """Deduplicate consecutive identical lines with efficient algorithm.
    
    Optimizations:
    - Single pass O(n)
    - Early termination on differences
    - Memory-efficient without extra copies
    """
    if not lines:
        return []
    
    if len(lines) == 1:
        return lines[:]
    
    out: List[str] = []
    i = 0
    
    while i < len(lines):
        cur = lines[i]
        rep = 1
        
        # Count consecutive repeats
        while i + rep < len(lines) and lines[i + rep] == cur:
            rep += 1
        
        # Add with repeat count if applicable
        if rep > 1:
            out.append(f"{cur} (repeated {rep}x)")
        else:
            out.append(cur)
        
        # Skip past all repeats
        i += rep
    
    return out


 


def _build_compact(transcript: str, max_lines: int) -> str:
    tx = (transcript or "").strip()
    if not tx:
        return ""
    raw = [ln.rstrip() for ln in tx.splitlines() if ln.strip()]
    if not raw:
        return ""
    # Build turn-based compact lines with clear speaker labels; condense multi-line turns
    out: List[str] = []
    cur_role: str | None = None  # 'User' | 'Jinx'
    buf: List[str] = []

    def _flush() -> None:
        nonlocal buf, cur_role
        if not buf:
            return
        text = " ".join(buf)
        text = _MULTI_SPACE_RE.sub(" ", text).strip()
        
        if not text:
            buf = []
            return
        
        if cur_role in ("User", "Jinx"):
            # Trim very long lines to keep memory lean
            try:
                lim = int(os.getenv("JINX_MEM_COMPACT_LINE_MAX_CHARS", "320"))
            except Exception:
                lim = 320
            
            # Remove stray trailing role labels (use cached regex)
            text = _TRAILING_ROLE_RE.sub("", text)
            
            # Truncate to limit
            if len(text) > lim:
                text = text[:lim]
            
            out.append(f"{cur_role}: {text}")
        else:
            # Fallback: treat as note
            out.append(f"Note: {text}")
        buf = []

    for ln in raw:
        # Robust label parse (leading spaces allowed) - use cached regex
        m = _ROLE_RE.match(ln)
        if m:
            label = m.group(1).strip().lower()
            val = m.group(2).strip() if m.group(2) else ""
            
            if label in ("user", "jinx"):
                _flush()
                cur_role = "User" if label == "user" else "Jinx"
                buf = [val] if val else []
                continue
            else:
                _flush()
                title = label.capitalize()
                out.append(f"{title}: {val}" if val else f"{title}:")
                cur_role = None
                buf = []
                continue
        # Continuation or role inference
        if cur_role:
            if cur_role == "User" and _is_code_like_line(ln):
                # Switch to agent output if code-like content follows the user line
                _flush()
                cur_role = "Jinx"
                buf = [ln]
            else:
                buf.append(ln)
        else:
            # Unknown provenance — classify by heuristics
            if _is_code_like_line(ln):
                out.append(f"Jinx: {ln[:320]}")
            else:
                out.append(f"Note: {ln[:320]}")
    _flush()

    # Keep only the most recent max_lines, with simple consecutive dedupe
    out = out[-(max_lines * 2) :]
    out = _dedupe_consecutive(out)[-max_lines:]
    return _redact_text("\n".join(out))


def _topn(counter: Dict[str, float], n: int) -> List[str]:
    items = sorted(counter.items(), key=lambda x: (-float(x[1]), x[0]))
    return [k for k, _ in items[:n]]


 


def _extract_settings(text: str) -> List[str]:
    out: List[str] = []
    for m in _SET_RE.finditer(text or ""):
        k = (m.group(1) or "").strip()
        v = (m.group(2) or "").strip()
        if k and v:
            out.append(f"setting: {k}=<redacted>")
    return out


def _redact_line(s: str) -> str:
    if not s:
        return s
    try:
        s2 = _EMAIL_RE.sub("<email>", s)
    except Exception:
        s2 = s
    try:
        s2 = _TOKEN_RE.sub("<token>", s2)
    except Exception:
        pass
    return s2


def _redact_text(text: str) -> str:
    out: List[str] = []
    for ln in (text or "").splitlines():
        out.append(_redact_line(ln))
    return "\n".join(out)


def _build_evergreen(transcript: str, evergreen_prev: str, facts_cap: int, top_paths: int, top_symbols: int, top_prefs: int, top_decs: int) -> str:
    text = (transcript or "")
    prev_lines = [ln.strip() for ln in (evergreen_prev or "").splitlines() if ln.strip()]
    prev_set = set(prev_lines)

    # Load persistent facts
    facts = _facts_load()

    # Temporal decay
    now_ms = int(time.time() * 1000)
    try:
        half_days = float(os.getenv("JINX_MEM_HALF_LIFE_DAYS", "7"))
    except Exception:
        half_days = 7.0
    try:
        stale_drop = float(os.getenv("JINX_MEM_DROP_THRESHOLD", "0.6"))
    except Exception:
        stale_drop = 0.6
    last_ts = int(facts.get("last_update_ts") or 0)
    if last_ts > 0 and half_days > 0:
        decay = 0.5 ** (max(0.0, (now_ms - last_ts) / 1000.0) / (half_days * 86400.0))
        # apply decay to all numeric maps
        for key in ("paths", "symbols", "prefs", "decisions"):
            bucket = facts.get(key) or {}
            new_bucket: Dict[str, float] = {}
            if isinstance(bucket, dict):
                for k, v in bucket.items():
                    try:
                        nv = float(v) * float(decay)
                    except Exception:
                        nv = 0.0
                    if nv >= stale_drop:
                        new_bucket[k] = nv
            facts[key] = new_bucket

    # Paths
    paths: Dict[str, float] = dict(facts.get("paths", {}))
    for m in _PATH_RE.finditer(text):
        p = (m.group(0) or "").strip()
        if p:
            paths[p] = float(paths.get(p, 0.0)) + 1.0
    facts["paths"] = paths

    # Symbols (identifiers)
    symbols: Dict[str, float] = dict(facts.get("symbols", {}))
    try:
        for tok in extract_identifiers(text, max_items=256):
            tl = tok.strip()
            if tl:
                symbols[tl] = float(symbols.get(tl, 0.0)) + 1.0
    except Exception:
        pass
    facts["symbols"] = symbols

    # Preferences
    prefs: Dict[str, float] = dict(facts.get("prefs", {}))
    for frag in _extract_pref_frags(text):
        prefs[frag] = float(prefs.get(frag, 0.0)) + 1.0
    facts["prefs"] = prefs

    # Decisions
    decs: Dict[str, float] = dict(facts.get("decisions", {}))
    for frag in _extract_decision_frags(text):
        decs[frag] = float(decs.get(frag, 0.0)) + 1.0
    facts["decisions"] = decs

    # Apply usage-based boost prior to capping (reinforce frequently used facts)
    try:
        _use_boost = str(os.getenv("JINX_MEM_USAGE_BOOST_EVG", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        _use_boost = True
    if _use_boost:
        def _apply_usage_boost(bucket: Dict[str, float], prefix: str) -> Dict[str, float]:
            if not isinstance(bucket, dict):
                return {}
            out: Dict[str, float] = {}
            for k, v in bucket.items():
                line = f"{prefix}{k}" if prefix else str(k)
                try:
                    w = float(_usage_weight(line))
                except Exception:
                    w = 1.0
                try:
                    out[k] = float(v) * max(0.5, min(2.0, w))
                except Exception:
                    out[k] = float(v) if isinstance(v, (int, float)) else 0.0
            return out

        facts["paths"] = _apply_usage_boost(facts.get("paths", {}), "path: ")
        facts["symbols"] = _apply_usage_boost(facts.get("symbols", {}), "symbol: ")
        facts["prefs"] = _apply_usage_boost(facts.get("prefs", {}), "pref: ")
        facts["decisions"] = _apply_usage_boost(facts.get("decisions", {}), "decision: ")

    # Drop low-weight facts prior to capping (promotion threshold)
    try:
        min_w = float(os.getenv("JINX_MEM_FACT_MIN_WEIGHT", "1.0"))
    except Exception:
        min_w = 1.0
    if min_w > 0.0:
        def _drop_low(d: Dict[str, float]) -> Dict[str, float]:
            if not isinstance(d, dict):
                return {}
            return {k: float(v) for k, v in d.items() if float(v or 0.0) >= min_w}
        facts["paths"] = _drop_low(facts.get("paths", {}))
        facts["symbols"] = _drop_low(facts.get("symbols", {}))
        facts["prefs"] = _drop_low(facts.get("prefs", {}))
        facts["decisions"] = _drop_low(facts.get("decisions", {}))

    # Age-based demotion: drop items not used in the last X days unless count >= keep threshold
    try:
        demote_days = float(os.getenv("JINX_MEM_DEMOTE_AGE_DAYS", "0"))
    except Exception:
        demote_days = 0.0
    try:
        keep_old_min = int(os.getenv("JINX_MEM_KEEP_OLD_MIN_COUNT", "2"))
    except Exception:
        keep_old_min = 2
    if demote_days > 0.0:
        cutoff = int((demote_days * 86400.0) * 1000.0)
        now_ms = int(time.time() * 1000)
        def _drop_old_age(bucket: Dict[str, float], prefix: str) -> Dict[str, float]:
            if not isinstance(bucket, dict):
                return {}
            out: Dict[str, float] = {}
            for k, v in bucket.items():
                line = f"{prefix}{k}" if prefix else str(k)
                try:
                    ts = int(_last_used_ms(line))
                except Exception:
                    ts = 0
                try:
                    cnt = int(_usage_count(line))
                except Exception:
                    cnt = 0
                # Keep if we have no ts info, or recent, or count above threshold
                if ts <= 0 or (now_ms - ts) <= cutoff or cnt >= keep_old_min:
                    out[k] = v
            return out
        facts["paths"] = _drop_old_age(facts.get("paths", {}), "path: ")
        facts["symbols"] = _drop_old_age(facts.get("symbols", {}), "symbol: ")
        facts["prefs"] = _drop_old_age(facts.get("prefs", {}), "pref: ")
        facts["decisions"] = _drop_old_age(facts.get("decisions", {}), "decision: ")

    # Apply cap
    def _cap(d: Dict[str, float]) -> Dict[str, float]:
        if len(d) <= facts_cap:
            return d
        keep = {k: d[k] for k in _topn(d, facts_cap)}
        return keep

    facts["paths"] = _cap(facts.get("paths", {}))
    facts["symbols"] = _cap(facts.get("symbols", {}))
    facts["prefs"] = _cap(facts.get("prefs", {}))
    facts["decisions"] = _cap(facts.get("decisions", {}))

    # Persist facts
    facts["last_update_ts"] = now_ms
    _facts_save(facts)

    # Render evergreen lines (rebuild from facts, carry previous explicit settings, include pins)
    lines: List[str] = []
    # Pinned lines first (redacted)
    try:
        _pins = _load_pins()
    except Exception:
        _pins = []
    pin_lines = [_redact_line(x) for x in _pins if x]
    # Soft-boost pinned facts in weights (best-effort)
    for pl in pin_lines:
        low = pl.lower()
        if low.startswith("path: "):
            k = pl[6:].strip()
            if k:
                facts.setdefault("paths", {})[k] = float(facts.get("paths", {}).get(k, 0.0)) + 2.0
        elif low.startswith("symbol: "):
            k = pl[8:].strip()
            if k:
                facts.setdefault("symbols", {})[k] = float(facts.get("symbols", {}).get(k, 0.0)) + 2.0
        elif low.startswith("pref: "):
            k = pl[6:].strip()
            if k:
                facts.setdefault("prefs", {})[k] = float(facts.get("prefs", {}).get(k, 0.0)) + 2.0
        elif low.startswith("decision: "):
            k = pl[10:].strip()
            if k:
                facts.setdefault("decisions", {})[k] = float(facts.get("decisions", {}).get(k, 0.0)) + 2.0
    # Start with pins list
    lines.extend(pin_lines)
    for p in _topn(facts.get("paths", {}), top_paths):
        lines.append(f"path: {p}")
    for s in _topn(facts.get("symbols", {}), top_symbols):
        lines.append(f"symbol: {s}")
    for pr in _topn(facts.get("prefs", {}), top_prefs):
        lines.append(f"pref: {pr}")
    for dec in _topn(facts.get("decisions", {}), top_decs):
        lines.append(f"decision: {dec}")

    # Settings gleaned from current transcript
    cur_settings = _extract_settings(text)
    for st in cur_settings:
        lines.append(st)

    # Carry previous explicit settings if absent this round
    prev_settings = [ln for ln in prev_lines if ln.lower().startswith("setting: ")]
    for st in prev_settings:
        if st not in lines:
            lines.append(st)

    # Optional max lines cap
    try:
        max_lines = int(os.getenv("JINX_MEM_EVERGREEN_MAX_LINES", "160"))
    except Exception:
        max_lines = 160
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]

    return _redact_text("\n".join(lines))


def build_local_memory(transcript: str | None, evergreen_prev: str | None, token_hint: int | None = None) -> Tuple[str, str | None]:
    """Local, rule-based memory builder.

    Returns (compact, evergreen) strings. Evergreen may be None if no update is needed.
    """
    t = (transcript or "").strip()
    eprev = (evergreen_prev or "").strip()
    # Strip tool blocks to avoid polluting memory with execution tags
    t_clean, _blocks = _strip_tool_blocks(t)

    # Budgets
    try:
        compact_lines = max(24, int(os.getenv("JINX_MEM_COMPACT_MAX_LINES", "200")))
    except Exception:
        compact_lines = 200
    try:
        facts_cap = max(50, int(os.getenv("JINX_MEM_FACTS_CAP", "800")))
    except Exception:
        facts_cap = 800
    try:
        top_paths = max(8, int(os.getenv("JINX_MEM_TOP_PATHS", "60")))
    except Exception:
        top_paths = 60
    try:
        top_symbols = max(8, int(os.getenv("JINX_MEM_TOP_SYMBOLS", "80")))
    except Exception:
        top_symbols = 80
    try:
        top_prefs = max(4, int(os.getenv("JINX_MEM_TOP_PREFS", "40")))
    except Exception:
        top_prefs = 40
    try:
        top_decs = max(4, int(os.getenv("JINX_MEM_TOP_DECS", "40")))
    except Exception:
        top_decs = 40

    # Dynamic budgets based on recent token usage (optional, default ON)
    try:
        dyn_on = str(os.getenv("JINX_MEM_DYNAMIC_BUDGET", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        dyn_on = True
    if dyn_on and (token_hint or 0) > 0:
        th = max(0, int(token_hint or 0))
        try:
            tgt = int(os.getenv("JINX_MEM_DYNAMIC_TARGET", "7000"))
        except Exception:
            tgt = 7000
        # Scale in [0.6, 1.2], shrink when tokens>>target, expand when small
        import math
        ratio = (th - tgt) / max(1000.0, float(tgt))
        scale = 1.0 - 0.4 * math.tanh(ratio)  # ~0.6..1.4 but clamp to 1.2
        if scale > 1.2:
            scale = 1.2
        if scale < 0.6:
            scale = 0.6
        # Apply to compact lines and facts caps/topNs proportionally
        compact_lines = max(24, int(compact_lines * scale))
        facts_cap = max(200, int(facts_cap * scale))
        top_paths = max(8, int(top_paths * scale))
        top_symbols = max(8, int(top_symbols * scale))
        top_prefs = max(4, int(top_prefs * scale))
        top_decs = max(4, int(top_decs * scale))

    compact = _build_compact(t_clean, max_lines=compact_lines)
    evergreen = _build_evergreen(t_clean, eprev, facts_cap, top_paths, top_symbols, top_prefs, top_decs)

    # Avoid unnecessary evergreen writes if unchanged
    evg_norm = ensure_nl(evergreen).rstrip("\n") if evergreen else ""
    prev_norm = ensure_nl(eprev).rstrip("\n") if eprev else ""
    durable_out = evg_norm if evg_norm and (evg_norm != prev_norm) else None

    # Normalize newlines
    return ensure_nl(compact).rstrip("\n"), durable_out
