"""Advanced code formatting chain with error recovery and metrics."""

from __future__ import annotations

from typing import List, Callable, Optional
from dataclasses import dataclass
import time

from .ast_normalize import ast_normalize
from .cst_format import cst_format
from .pep8_format import pep8_format
from .black_format import black_format


@dataclass
class FormatterMetrics:
    """Track formatter performance and errors."""
    total_calls: int = 0
    ast_normalize_errors: int = 0
    cst_format_errors: int = 0
    pep8_format_errors: int = 0
    black_format_errors: int = 0
    avg_format_time_ms: float = 0.0


_metrics = FormatterMetrics()


def chain_format(code: str) -> str:
    """Run a chain of formatters with best-effort error recovery.
    
    Each formatter is isolated - if one fails, the chain continues
    with the last successful output. This ensures we always return
    valid (even if not perfectly formatted) code.
    
    Args:
        code: Source code to format
    
    Returns:
        Formatted code (best-effort)
    """
    global _metrics
    
    t0 = time.perf_counter()
    _metrics.total_calls += 1
    
    x = code
    
    # Stage 1: AST normalize
    try:
        x = ast_normalize(x)
    except Exception:
        _metrics.ast_normalize_errors += 1
        # Continue with original code
    
    # Stage 2: CST format
    try:
        x = cst_format(x)
    except Exception:
        _metrics.cst_format_errors += 1
        # Continue with current x
    
    # Stage 3: PEP8 format
    try:
        x = pep8_format(x)
    except Exception:
        _metrics.pep8_format_errors += 1
        # Continue with current x
    
    # Stage 4: Black format
    try:
        x = black_format(x)
    except Exception:
        _metrics.black_format_errors += 1
        # Continue with current x
    
    # Update metrics
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    count = _metrics.total_calls
    _metrics.avg_format_time_ms = (
        (_metrics.avg_format_time_ms * (count - 1) + elapsed_ms) / count
    )
    
    return x


def get_formatter_metrics() -> FormatterMetrics:
    """Get formatting performance metrics."""
    return _metrics
