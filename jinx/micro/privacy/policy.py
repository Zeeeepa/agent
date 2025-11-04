from __future__ import annotations

import os
import re
from typing import Iterable, Pattern, List, Dict
from functools import lru_cache
from dataclasses import dataclass, field
import threading

# Defaults: STRICT privacy
# - internal filtering ON
# - absolute path redaction ON
# - PII redaction ON


def _is_on(env: str | None, default: bool = True) -> bool:
    val = (env if env is not None else ("1" if default else "0")).strip().lower()
    return val not in ("", "0", "false", "off", "no")


def filter_internals_enabled() -> bool:
    return _is_on(os.getenv("JINX_FILTER_INTERNALS"), True)


def filter_mode() -> str:
    m = (os.getenv("JINX_FILTER_MODE", "strip") or "strip").strip().lower()
    return "redact" if m == "redact" else "strip"


def restrict_abs_paths_enabled() -> bool:
    return _is_on(os.getenv("JINX_PRIVACY_ALLOW_ABS_PATHS"), False) is False


def pii_redact_enabled() -> bool:
    return _is_on(os.getenv("JINX_PRIVACY_PII_REDACT"), True)


@dataclass
class PIIPattern:
    """Configurable PII pattern with metadata."""
    name: str
    pattern: str
    sensitivity: str = "high"  # high, medium, low
    enabled: bool = True


class PIIDetector:
    """Advanced PII detection with extensible patterns and caching."""
    
    _instance: 'PIIDetector | None' = None
    _lock = threading.RLock()
    
    def __init__(self):
        self._patterns: Dict[str, Pattern[str]] = {}
        self._pattern_metadata: Dict[str, PIIPattern] = {}
        self._setup_default_patterns()
    
    @classmethod
    def get_instance(cls) -> 'PIIDetector':
        """Thread-safe singleton access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _setup_default_patterns(self) -> None:
        """Initialize default PII patterns with metadata."""
        default_patterns = [
            PIIPattern("openai_key", r"sk-[a-zA-Z0-9]{20,100}", "high"),
            PIIPattern("github_token", r"gh[pousr]_[A-Za-z0-9]{20,100}", "high"),
            PIIPattern("aws_access_key", r"AKIA[0-9A-Z]{16}", "high"),
            PIIPattern("slack_token", r"xox[abpr]-[A-Za-z0-9\-]{10,100}", "high"),
            PIIPattern("jwt_token", r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}", "high"),
            # Additional modern API keys
            PIIPattern("azure_key", r"[A-Za-z0-9+/]{88}==", "high"),
            PIIPattern("stripe_key", r"sk_(?:live|test)_[A-Za-z0-9]{24,99}", "high"),
            PIIPattern("google_api_key", r"AIza[A-Za-z0-9_\-]{35}", "high"),
            PIIPattern("generic_api_key", r"api[_-]?key[_-]?[=:][\s]*['\"][A-Za-z0-9_\-]{20,}['\"]" , "medium"),
            # Email addresses (medium sensitivity - context dependent)
            PIIPattern("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "medium"),
        ]
        
        for pii_pattern in default_patterns:
            self._add_pattern(pii_pattern)
    
    def _add_pattern(self, pii_pattern: PIIPattern) -> None:
        """Add a PII pattern with compilation."""
        if not pii_pattern.enabled:
            return
        
        try:
            compiled = re.compile(pii_pattern.pattern, re.IGNORECASE)
            self._patterns[pii_pattern.name] = compiled
            self._pattern_metadata[pii_pattern.name] = pii_pattern
        except re.error:
            # Skip invalid patterns
            pass
    
    def add_custom_pattern(self, pii_pattern: PIIPattern) -> None:
        """Add custom PII pattern at runtime."""
        with self._lock:
            self._add_pattern(pii_pattern)
    
    def get_patterns(self, min_sensitivity: str = "low") -> List[Pattern[str]]:
        """Get compiled patterns filtered by sensitivity level."""
        sensitivity_order = {"low": 0, "medium": 1, "high": 2}
        min_level = sensitivity_order.get(min_sensitivity, 0)
        
        result = []
        for name, pattern in self._patterns.items():
            metadata = self._pattern_metadata.get(name)
            if metadata and sensitivity_order.get(metadata.sensitivity, 0) >= min_level:
                result.append(pattern)
        
        return result
    
    @lru_cache(maxsize=1024)
    def redact(self, text: str, sensitivity: str = "medium") -> str:
        """Redact PII from text with caching."""
        if not text:
            return text
        
        out = text
        for pat in self.get_patterns(sensitivity):
            out = pat.sub("[REDACTED]", out)
        
        return out


# Global detector instance
_pii_detector = PIIDetector.get_instance()

# Legacy compatibility: expose as list
_PII_PATTERNS: List[Pattern[str]] = _pii_detector.get_patterns("low")


def pii_patterns() -> Iterable[Pattern[str]]:
    """Get all PII patterns (legacy interface)."""
    return _pii_detector.get_patterns("low")


def redact_pii(text: str) -> str:
    """Redact PII from text using advanced detector."""
    return _pii_detector.redact(text, "medium")


# Absolute path detectors (best-effort)
_WIN_ABS = re.compile(r"(?i)\b[A-Z]:\\")
_POSIX_ABS = re.compile(r"(^|\s)/(?:[^ \t\r\n]{1,256})")


def has_abs_path(s: str) -> bool:
    if not s:
        return False
    return bool(_WIN_ABS.search(s) or _POSIX_ABS.search(s))
