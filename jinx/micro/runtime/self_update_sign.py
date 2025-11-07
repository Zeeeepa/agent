from __future__ import annotations

import json
import os
from typing import Optional

from .self_update_journal import append as _jr


def _find_sign_file(stage_dir: str) -> Optional[str]:
    # Preferred: stage_dir/.jinx/sign.json, fallback: stage_dir/jinx/.sign.json
    cand1 = os.path.join(stage_dir, ".jinx", "sign.json")
    if os.path.isfile(cand1):
        return cand1
    cand2 = os.path.join(stage_dir, "jinx", ".sign.json")
    if os.path.isfile(cand2):
        return cand2
    return None


def verify_stage_signature(stage_dir: str, sha_calc: str) -> bool:
    """Verify staged package against provided signature file if present.
    - If no signature file, returns True unless REQUIRE_SIGNATURE=1.
    - If signature exists, compare sha256; on mismatch -> False.
    """
    req = str(os.getenv("JINX_SELFUPDATE_REQUIRE_SIGNATURE", "0")).lower() not in ("", "0", "false", "off", "no")
    sf = _find_sign_file(stage_dir)
    if not sf:
        if req:
            _jr("selfupdate.sign_missing", stage="stage", ok=False)
            return False
        _jr("selfupdate.sign_absent", stage="stage", ok=True)
        return True
    try:
        with open(sf, "r", encoding="utf-8") as f:
            data = json.load(f)
        sig_sha = str(data.get("sha256") or "").strip().lower()
        ok = bool(sig_sha) and (sig_sha == (sha_calc or "").strip().lower())
        _jr("selfupdate.sign_ok" if ok else "selfupdate.sign_mismatch", stage="stage", ok=ok)
        return ok
    except Exception:
        _jr("selfupdate.sign_error", stage="stage", ok=False)
        return not req
