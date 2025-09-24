from __future__ import annotations

import os

# Environment-driven tunables for project retrieval
PROJ_DEFAULT_TOP_K = int(os.getenv("EMBED_PROJECT_TOP_K", "6"))
PROJ_SCORE_THRESHOLD = float(os.getenv("EMBED_PROJECT_SCORE_THRESHOLD", "0.22"))
PROJ_MIN_PREVIEW_LEN = int(os.getenv("EMBED_PROJECT_MIN_PREVIEW_LEN", "12"))
PROJ_MAX_FILES = int(os.getenv("EMBED_PROJECT_MAX_FILES", "2000"))
PROJ_MAX_CHUNKS_PER_FILE = int(os.getenv("EMBED_PROJECT_MAX_CHUNKS_PER_FILE", "300"))
PROJ_QUERY_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Snippet shaping
PROJ_SNIPPET_AROUND = int(os.getenv("EMBED_PROJECT_SNIPPET_AROUND", "12"))
PROJ_SNIPPET_PER_HIT_CHARS = int(os.getenv("EMBED_PROJECT_SNIPPET_PER_HIT_CHARS", "900"))

# Always include full Python function/class scope when possible
def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() not in {"0", "false", "no", "off"}

PROJ_ALWAYS_FULL_PY_SCOPE = _env_bool("EMBED_PROJECT_ALWAYS_FULL_PY_SCOPE", True)
PROJ_SCOPE_MAX_CHARS = int(os.getenv("EMBED_PROJECT_SCOPE_MAX_CHARS", "0"))

# Overall budget for <embeddings_code> text (sum of all snippets)
PROJ_TOTAL_CODE_BUDGET = int(os.getenv("EMBED_PROJECT_TOTAL_CODE_BUDGET", "20000"))

# Limit the number of hits that expand to full Python scope; others will use windowed snippets (<=0 = unlimited)
PROJ_FULL_SCOPE_TOP_N = int(os.getenv("EMBED_PROJECT_FULL_SCOPE_TOP_N", "0"))
