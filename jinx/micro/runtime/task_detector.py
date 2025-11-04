"""Task Detector - Pure embedding-based task type detection (LANGUAGE-AGNOSTIC).

COMPLETELY REWRITTEN - NO KEYWORDS, NO REGEX, NO LANGUAGE DEPENDENCIES.

Uses:
- Pure embedding similarity
- Vector intent analysis
- Metric learning
- Prototype networks
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional


class TaskDetector:
    """
    Pure vector-based task detector wrapper.
    
    All keyword/regex matching REMOVED.
    Now delegates to embedding_task_classifier.py for pure vector-based detection.
    """
    
    # Task types (no patterns - pure embedding-based)
    TASK_TYPES = [
        'code_search', 'code_analysis', 'debugging', 'refactoring',
        'implementation', 'testing', 'documentation', 'planning', 'conversation'
    ]
    
    def __init__(self):
        pass  # No patterns to compile
    
    async def detect_with_embeddings(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """Pure embedding-based detection (NO KEYWORDS/REGEX)."""
        
        try:
            # Use pure ML classifier
            from jinx.micro.runtime.ml_task_classifier import get_ml_classifier
            
            ml_classifier = await get_ml_classifier()
            ml_type, ml_conf, ml_scores = await ml_classifier.predict(text, context)
            
            return (ml_type, ml_conf)
        
        except Exception:
            # Fallback
            return ('conversation', 0.5)
    
    def get_config_profile(self, task_type: str) -> Dict[str, str]:
        """Get recommended configuration profile for task type."""
        
        profiles = {
            'code_search': {
                'EMBED_PROJECT_TOP_K': '100',  # Many results for search
                'EMBED_PROJECT_EXHAUSTIVE': '1',  # All stages
                'JINX_STAGE_PROJCTX_MS': '8000',  # Extra time
                'EMBED_PROJECT_SCORE_THRESHOLD': '0.10',  # Very low threshold
                'JINX_MAX_CONCURRENT': '8',  # High concurrency
            },
            'code_analysis': {
                'EMBED_PROJECT_TOP_K': '50',
                'EMBED_PROJECT_EXHAUSTIVE': '1',
                'JINX_STAGE_PROJCTX_MS': '5000',
                'EMBED_PROJECT_CALLGRAPH': '1',  # Need relationships
                'EMBED_PROJECT_ALWAYS_FULL_PY_SCOPE': '1',  # Full code
                'JINX_MAX_CONCURRENT': '6',
            },
            'debugging': {
                'EMBED_PROJECT_TOP_K': '80',
                'EMBED_PROJECT_EXHAUSTIVE': '1',
                'JINX_STAGE_PROJCTX_MS': '7000',
                'EMBED_PROJECT_CALLGRAPH': '1',  # Need call chains
                'JINX_MAX_CONCURRENT': '4',  # Lower to avoid race conditions
                'EMBED_PROJECT_CALLGRAPH_CALLERS_LIMIT': '8',  # Deep traces
            },
            'refactoring': {
                'EMBED_PROJECT_TOP_K': '60',
                'EMBED_PROJECT_EXHAUSTIVE': '1',
                'JINX_STAGE_PROJCTX_MS': '6000',
                'EMBED_PROJECT_CALLGRAPH': '1',
                'EMBED_PROJECT_USAGE_REFS_LIMIT': '10',  # Find all usages
                'JINX_MAX_CONCURRENT': '5',
            },
            'implementation': {
                'EMBED_PROJECT_TOP_K': '40',
                'EMBED_PROJECT_EXHAUSTIVE': '0',  # Faster
                'JINX_STAGE_PROJCTX_MS': '3000',
                'EMBED_PROJECT_SCORE_THRESHOLD': '0.20',
                'JINX_MAX_CONCURRENT': '8',  # Fast iteration
            },
            'testing': {
                'EMBED_PROJECT_TOP_K': '30',
                'EMBED_PROJECT_EXHAUSTIVE': '0',
                'JINX_STAGE_PROJCTX_MS': '2000',
                'JINX_MAX_CONCURRENT': '6',
            },
            'documentation': {
                'EMBED_PROJECT_TOP_K': '20',
                'EMBED_PROJECT_EXHAUSTIVE': '0',
                'JINX_STAGE_PROJCTX_MS': '2000',
                'EMBED_PROJECT_ALWAYS_FULL_PY_SCOPE': '1',
                'JINX_MAX_CONCURRENT': '4',
            },
            'planning': {
                'EMBED_PROJECT_TOP_K': '30',
                'EMBED_PROJECT_EXHAUSTIVE': '0',
                'JINX_STAGE_PROJCTX_MS': '3000',
                'JINX_MAX_CONCURRENT': '5',
            },
            'conversation': {
                'EMBED_PROJECT_TOP_K': '20',
                'EMBED_PROJECT_EXHAUSTIVE': '0',
                'JINX_STAGE_PROJCTX_MS': '2000',
                'JINX_MAX_CONCURRENT': '4',
            }
        }
        
        return profiles.get(task_type, profiles['conversation'])


# Singleton
_task_detector: Optional[TaskDetector] = None


def get_task_detector() -> TaskDetector:
    """Get singleton task detector."""
    global _task_detector
    if _task_detector is None:
        _task_detector = TaskDetector()
    return _task_detector


async def detect_task_type(
    text: str,
    context: Optional[Dict] = None,
    use_embeddings: bool = True
) -> Tuple[str, float]:
    """Detect task type with optional embeddings."""
    detector = get_task_detector()
    
    if use_embeddings:
        return await detector.detect_with_embeddings(text, context)
    else:
        return detector.detect(text, context)


def get_optimal_config_for_task(task_type: str) -> Dict[str, str]:
    """Get optimal configuration for task type."""
    detector = get_task_detector()
    return detector.get_config_profile(task_type)


__all__ = [
    "TaskDetector",
    "get_task_detector",
    "detect_task_type",
    "get_optimal_config_for_task",
]
