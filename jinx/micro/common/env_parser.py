"""Centralized environment variable parsing with validation and caching.

This module provides advanced environment variable parsing utilities that:
- Cache parsed values to avoid repeated parsing
- Validate and clamp values to safe ranges
- Provide type-safe accessors
- Support default values with fallback chains
"""

from __future__ import annotations

import os
from typing import TypeVar, Callable, Optional, Dict, Any
from functools import lru_cache
import threading

T = TypeVar('T')

# Thread-safe cache for parsed environment variables
_env_cache: Dict[str, Any] = {}
_env_cache_lock = threading.RLock()


def _is_truthy(val: Optional[str]) -> bool:
    """Check if string value represents a truthy boolean."""
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def _is_falsy(val: Optional[str]) -> bool:
    """Check if string value represents a falsy boolean."""
    if not val:
        return True
    return val.strip().lower() in {"", "0", "false", "no", "off", "disable", "disabled"}


@lru_cache(maxsize=256)
def get_bool(key: str, default: bool = False, *, cache: bool = True) -> bool:
    """Parse environment variable as boolean with caching.
    
    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        cache: Whether to cache the result (default: True)
    
    Returns:
        Parsed boolean value
    
    Examples:
        >>> get_bool("JINX_FEATURE_ENABLE", default=True)
        True
        >>> get_bool("JINX_DEBUG", default=False)
        False
    """
    try:
        val = os.getenv(key)
        if val is None:
            return default
        
        if _is_truthy(val):
            return True
        if _is_falsy(val):
            return False
        
        return default
    except Exception:
        return default


@lru_cache(maxsize=256)
def get_int(
    key: str, 
    default: int = 0, 
    *, 
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    cache: bool = True
) -> int:
    """Parse environment variable as integer with optional clamping.
    
    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        cache: Whether to cache the result (default: True)
    
    Returns:
        Parsed and clamped integer value
    
    Examples:
        >>> get_int("JINX_MAX_WORKERS", default=4, min_val=1, max_val=64)
        4
        >>> get_int("JINX_TIMEOUT", default=30, min_val=5)
        30
    """
    try:
        val = os.getenv(key)
        if val is None:
            result = default
        else:
            result = int(val)
        
        # Apply clamping if specified
        if min_val is not None and result < min_val:
            result = min_val
        if max_val is not None and result > max_val:
            result = max_val
        
        return result
    except (ValueError, TypeError):
        return default


@lru_cache(maxsize=256)
def get_float(
    key: str,
    default: float = 0.0,
    *,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    cache: bool = True
) -> float:
    """Parse environment variable as float with optional clamping.
    
    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        cache: Whether to cache the result (default: True)
    
    Returns:
        Parsed and clamped float value
    
    Examples:
        >>> get_float("JINX_ALPHA", default=0.3, min_val=0.0, max_val=1.0)
        0.3
    """
    try:
        val = os.getenv(key)
        if val is None:
            result = default
        else:
            result = float(val)
        
        # Apply clamping if specified
        if min_val is not None and result < min_val:
            result = min_val
        if max_val is not None and result > max_val:
            result = max_val
        
        return result
    except (ValueError, TypeError):
        return default


@lru_cache(maxsize=256)
def get_str(key: str, default: str = "", *, strip: bool = True) -> str:
    """Parse environment variable as string.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        strip: Whether to strip whitespace (default: True)
    
    Returns:
        String value
    """
    try:
        val = os.getenv(key)
        if val is None:
            return default
        return val.strip() if strip else val
    except Exception:
        return default


def get_list(
    key: str,
    default: Optional[list[str]] = None,
    *,
    separator: str = ",",
    strip: bool = True,
    unique: bool = False
) -> list[str]:
    """Parse environment variable as list of strings.
    
    Args:
        key: Environment variable name
        default: Default list if not set
        separator: Delimiter for splitting (default: comma)
        strip: Whether to strip whitespace from items
        unique: Whether to return only unique items
    
    Returns:
        List of string values
    
    Examples:
        >>> get_list("JINX_FEATURES", default=[], separator=",")
        ['feature1', 'feature2']
    """
    if default is None:
        default = []
    
    try:
        val = os.getenv(key)
        if not val:
            return default
        
        items = [item.strip() if strip else item for item in val.split(separator)]
        items = [item for item in items if item]  # Remove empty strings
        
        if unique:
            seen = set()
            unique_items = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)
            return unique_items
        
        return items
    except Exception:
        return default


def clear_cache() -> None:
    """Clear all cached environment variable values.
    
    This should be called if environment variables change at runtime.
    """
    get_bool.cache_clear()
    get_int.cache_clear()
    get_float.cache_clear()
    get_str.cache_clear()
    
    with _env_cache_lock:
        _env_cache.clear()


def get_or_compute(
    key: str,
    compute: Callable[[], T],
    default: Optional[T] = None,
    *,
    cache: bool = True
) -> T:
    """Get environment value or compute it using a callback.
    
    Args:
        key: Cache key (not necessarily an env var)
        compute: Function to compute value if not cached
        default: Default value if computation fails
        cache: Whether to cache the computed result
    
    Returns:
        Cached or computed value
    """
    if cache:
        with _env_cache_lock:
            if key in _env_cache:
                return _env_cache[key]
    
    try:
        result = compute()
        if cache:
            with _env_cache_lock:
                _env_cache[key] = result
        return result
    except Exception:
        return default if default is not None else compute()


# Convenience aliases for common patterns
def is_enabled(key: str, default: bool = False) -> bool:
    """Check if a feature flag is enabled."""
    return get_bool(key, default)


def is_disabled(key: str, default: bool = True) -> bool:
    """Check if a feature flag is disabled."""
    return not get_bool(key, not default)


__all__ = [
    'get_bool',
    'get_int',
    'get_float',
    'get_str',
    'get_list',
    'get_or_compute',
    'is_enabled',
    'is_disabled',
    'clear_cache',
]
