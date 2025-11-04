from __future__ import annotations

import re
from typing import Optional, Dict, Set
from functools import lru_cache
import threading

# Language-agnostic turns detection.
# Returns {"kind": "user|jinx|pair", "index": int} or None.

_FULLWIDTH = str.maketrans({
    "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9"
})

class RoleMapper:
    """Extensible role token mapping with thread safety."""
    
    _instance: 'RoleMapper | None' = None
    _lock = threading.RLock()
    
    def __init__(self):
        self._role_map: Dict[str, str] = {}
        self._setup_default_mappings()
    
    @classmethod
    def get_instance(cls) -> 'RoleMapper':
        """Thread-safe singleton access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _setup_default_mappings(self) -> None:
        """Initialize default role token mappings."""
        # Assistant/Agent role tokens
        assistant_tokens = [
            "jinx", "assistant", "agent", "bot", "ai", 
            "cascade", "system", "model"
        ]
        for token in assistant_tokens:
            self._role_map[token] = "jinx"
        
        # User/Human role tokens
        user_tokens = [
            "user", "human", "operator", "person", 
            "me", "my", "you", "your"
        ]
        for token in user_tokens:
            self._role_map[token] = "user"
    
    def add_mapping(self, token: str, role: str) -> None:
        """Add custom role token mapping."""
        with self._lock:
            if role in {"jinx", "user", "pair"}:
                self._role_map[token.lower()] = role
    
    @lru_cache(maxsize=1024)
    def detect_role(self, text: str) -> str:
        """Detect role from text with caching."""
        if not text:
            return "pair"
        
        t = text.lower()
        
        # Check for role tokens with word boundary matching for accuracy
        for token, role in self._role_map.items():
            # Use word boundary regex for better precision
            if re.search(rf'\b{re.escape(token)}\b', t):
                return role
        
        # Default to pair (safer and still useful)
        return "pair"


# Global role mapper instance
_role_mapper = RoleMapper.get_instance()

# Legacy compatibility
_ROLE_TOKENS = _role_mapper._role_map.copy()

_NEAR_MSG_PATTERNS = [
    # Explicit markers
    re.compile(r"(?i)#\s*(\d{1,4})\b"),
    re.compile(r"(?i)(?:No\.?|№|N°)\s*(\d{1,4})\b"),
    # East Asian ordinal marker: 第 12
    re.compile(r"(?i)第\s*([０-９0-9]{1,4})\b"),
    # Generic: (msg|message|turn|step|сообщ|ход|шаг) 12
    re.compile(r"(?iu)\b(?:msg|message|turn|step|сообщ\w*|ход|шаг)\s*[:#\- ]*([０-９0-9]{1,4})\b"),
]

_ANY_INT = re.compile(r"(?i)\b([０-９0-9]{1,4})\b")

_ROMAN_RE = re.compile(r"\b([IVXLCDM]{1,6})\b", re.I)
_ROMAN_MAP = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}


def _roman_to_int(s: str) -> int:
    s = s.upper().strip()
    total = 0
    prev = 0
    for ch in reversed(s):
        v = _ROMAN_MAP.get(ch, 0)
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def _extract_index(text: str) -> Optional[int]:
    t = (text or "").strip()
    if not t:
        return None
    # 1) Try explicit nearby message markers
    for rx in _NEAR_MSG_PATTERNS:
        m = rx.search(t)
        if m:
            try:
                raw = (m.group(1) or "").translate(_FULLWIDTH)
                v = int(raw)
                return v if v > 0 else None
            except Exception:
                continue
    # 2) Try roman numerals (very conservative)
    m = _ROMAN_RE.search(t)
    if m:
        v = _roman_to_int(m.group(1) or "")
        if v > 0:
            return v
    # 3) Fallback: first integer token anywhere
    m = _ANY_INT.search(t)
    if m:
        try:
            raw = (m.group(1) or "").translate(_FULLWIDTH)
            v = int(raw)
            return v if v > 0 else None
        except Exception:
            pass
    return None


def _detect_kind(text: str) -> str:
    """Detect conversation role kind from text."""
    return _role_mapper.detect_role(text)


def detect_turn_query(text: str) -> Optional[Dict[str, object]]:
    idx = _extract_index(text)
    if not idx:
        return None
    kind = _detect_kind(text)
    return {"kind": kind, "index": int(idx)}


__all__ = ["detect_turn_query"]
