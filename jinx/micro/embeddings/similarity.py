from __future__ import annotations

from typing import Iterable, List

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore


def score_cosine_batch(query_vec: List[float], vectors: Iterable[List[float]]) -> List[float]:
    """Compute cosine similarities between query_vec and a batch of vectors.

    Uses NumPy when available for fast vectorized computation; otherwise falls back to
    a pure Python implementation.
    """
    q = list(query_vec or [])
    if not q:
        return [0.0 for _ in vectors]

    # Vectorized path
    if _np is not None:
        try:
            mat = _np.asarray(list(vectors), dtype=_np.float32)
            qv = _np.asarray(q, dtype=_np.float32)
            if mat.ndim != 2 or qv.ndim != 1 or mat.shape[1] != qv.shape[0]:
                # shape mismatch -> fallback
                raise ValueError("shape")
            qn = _np.linalg.norm(qv)
            if qn <= 0:
                return [0.0 for _ in range(mat.shape[0])]
            dn = _np.linalg.norm(mat, axis=1)
            # Avoid division by zero
            dn = _np.where(dn <= 0, 1.0, dn)
            sims = (mat @ qv) / (dn * qn)
            return [float(x) for x in sims.tolist()]
        except Exception:
            pass

    # Fallback: pure Python cosine
    out: List[float] = []
    import math as _math
    for v in vectors:
        if not v or len(v) != len(q):
            out.append(0.0)
            continue
        dot = 0.0
        na = 0.0
        nb = 0.0
        for a, b in zip(v, q):
            fa = float(a)
            fb = float(b)
            dot += fa * fb
            na += fa * fa
            nb += fb * fb
        if na <= 0.0 or nb <= 0.0:
            out.append(0.0)
            continue
        out.append(dot / (_math.sqrt(na) * _math.sqrt(nb)))
    return out
