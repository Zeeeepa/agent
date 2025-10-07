from __future__ import annotations

# Re-export public API for continuity micro-modules (cont/*)

from .util import is_short_followup
from .anchors import (
    extract_anchors,
    last_agent_question,
    last_user_query,
)
from .query import augment_query_for_retrieval
from .render import render_continuity_block
from .cache import (
    load_last_context,
    save_last_context,
    maybe_reuse_last_context,
    load_last_anchors,
)
from .meta import (
    load_cache_meta,
    load_cache_meta_sync,
    save_last_context_with_meta,
)
from .topic import detect_topic_shift
from .compactor import maybe_compact_state_frames

__all__ = [
    # util
    "is_short_followup",
    # anchors
    "extract_anchors",
    "last_agent_question",
    "last_user_query",
    # query/task
    "augment_query_for_retrieval",
    # render
    "render_continuity_block",
    # cache
    "load_last_context",
    "save_last_context",
    "maybe_reuse_last_context",
    "load_last_anchors",
    # meta
    "load_cache_meta",
    "load_cache_meta_sync",
    "save_last_context_with_meta",
    # topic
    "detect_topic_shift",
    # compactor
    "maybe_compact_state_frames",
]
