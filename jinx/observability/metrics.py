from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict

_METRICS_DIR = os.path.join(".jinx", "stats")
_METRICS_PATH = os.path.join(_METRICS_DIR, "patch_metrics.json")
_lock = threading.RLock()


def _ensure_dir() -> None:
    try:
        os.makedirs(_METRICS_DIR, exist_ok=True)
    except Exception:
        pass


def _load() -> Dict:
    try:
        with open(_METRICS_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    return {}


def _save(obj: Dict) -> None:
    _ensure_dir()
    try:
        with open(_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def record_patch_event(ctx: str, strategy: str, ok: bool, diff_total: int) -> None:
    """Record patch commit metrics: counts per context and strategy.
    ctx: bandit context string; strategy: chosen candidate name; ok: commit result; diff_total: additions+removals
    """
    with _lock:
        data = _load()
        root = data.setdefault("contexts", {})
        c = root.setdefault(ctx or "default", {"count": 0, "success": 0, "fail": 0, "avg_diff": 0.0, "last_ts": 0.0, "strategies": {}})
        c["count"] = int(c.get("count", 0)) + 1
        if ok:
            c["success"] = int(c.get("success", 0)) + 1
        else:
            c["fail"] = int(c.get("fail", 0)) + 1
        # ema for avg_diff
        try:
            ema = float(c.get("avg_diff", 0.0))
            a = 0.2
            c["avg_diff"] = a * float(diff_total or 0) + (1 - a) * ema
        except Exception:
            c["avg_diff"] = float(diff_total or 0)
        c["last_ts"] = _now()
        # per-strategy
        sroot = c.setdefault("strategies", {})
        s = sroot.setdefault(strategy, {"count": 0, "success": 0, "fail": 0, "avg_diff": 0.0, "last_ts": 0.0})
        s["count"] = int(s.get("count", 0)) + 1
        if ok:
            s["success"] = int(s.get("success", 0)) + 1
        else:
            s["fail"] = int(s.get("fail", 0)) + 1
        try:
            ema = float(s.get("avg_diff", 0.0))
            a = 0.2
            s["avg_diff"] = a * float(diff_total or 0) + (1 - a) * ema
        except Exception:
            s["avg_diff"] = float(diff_total or 0)
        s["last_ts"] = _now()
        _save(data)
    # OTEL export (best-effort, env-gated)
    try:
        if str(os.getenv("JINX_OTEL_PATCH_METRICS", "1")).lower() not in ("", "0", "false", "off", "no"):
            from jinx.observability.otel import span as _span
            attrs = {
                "ctx": ctx or "default",
                "strategy": strategy,
                "ok": bool(ok),
                "diff_total": int(diff_total or 0),
                # Aggregates are approximate here (post-save read avoided for RT):
            }
            with _span("autopatch.commit_metric", attrs=attrs):
                pass
    except Exception:
        pass
