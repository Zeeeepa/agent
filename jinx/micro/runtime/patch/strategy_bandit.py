from __future__ import annotations

import os
import json
import time
from typing import Dict, Tuple, List

_BANDIT_PATH = os.path.join(".jinx", "brain", "patch_bandit.json")
_OS_MAKEDIRS = os.makedirs


def _ensure_dir() -> None:
    try:
        _OS_MAKEDIRS(os.path.dirname(_BANDIT_PATH), exist_ok=True)
    except Exception:
        pass


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def _load() -> Dict[str, Dict[str, Dict[str, float]]]:
    try:
        with open(_BANDIT_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj  # type: ignore[return-value]
    except Exception:
        pass
    return {}


def _save(obj: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    _ensure_dir()
    try:
        with open(_BANDIT_PATH, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass


def _decay(value: float, last_ts: float, half_life_sec: float) -> float:
    try:
        dt = max(0.0, _now() - float(last_ts or 0.0))
        if half_life_sec <= 0.0:
            return value
        return value * (0.5 ** (dt / half_life_sec))
    except Exception:
        return value


def bandit_order_for_context(ctx: str, strategies: List[str]) -> List[str]:
    """Return strategies ordered by UCB-like score for a given context.

    Keeps exploration by giving unseen strategies a modest prior.
    """
    data = _load()
    sdata = data.get(ctx) or {}
    now = _now()
    try:
        hl = float(os.getenv("JINX_AUTOPATCH_BANDIT_HALF_SEC", "1800"))
    except Exception:
        hl = 1800.0
    out: List[Tuple[float, str]] = []
    # Global trial count for exploration term
    total_trials = 1.0
    for s in strategies:
        ent = sdata.get(s) or {"succ": 0.0, "fail": 0.0, "ts": now}
        total_trials += ent.get("succ", 0.0) + ent.get("fail", 0.0)
    for s in strategies:
        ent = sdata.get(s) or {"succ": 0.0, "fail": 0.0, "ts": now}
        succ = _decay(float(ent.get("succ", 0.0)), float(ent.get("ts", now)), hl)
        fail = _decay(float(ent.get("fail", 0.0)), float(ent.get("ts", now)), hl)
        trials = max(1.0, succ + fail)
        rate = succ / trials
        # UCB-like exploration bonus
        import math
        bonus = math.sqrt(max(0.0, math.log(total_trials) / trials))
        score = rate + 0.4 * bonus
        out.append((score, s))
    out.sort(key=lambda t: t[0], reverse=True)
    return [s for _sc, s in out]


def bandit_update(ctx: str, strategy: str, success: bool) -> None:
    data = _load()
    now = _now()
    sdata = data.setdefault(ctx, {})
    ent = sdata.get(strategy) or {"succ": 0.0, "fail": 0.0, "ts": now}
    # Decay existing counts to avoid stale history domination
    try:
        hl = float(os.getenv("JINX_AUTOPATCH_BANDIT_HALF_SEC", "1800"))
    except Exception:
        hl = 1800.0
    ent["succ"] = _decay(float(ent.get("succ", 0.0)), float(ent.get("ts", now)), hl)
    ent["fail"] = _decay(float(ent.get("fail", 0.0)), float(ent.get("ts", now)), hl)
    if success:
        ent["succ"] = float(ent.get("succ", 0.0)) + 1.0
    else:
        ent["fail"] = float(ent.get("fail", 0.0)) + 1.0
    ent["ts"] = now
    sdata[strategy] = ent
    data[ctx] = sdata
    _save(data)
