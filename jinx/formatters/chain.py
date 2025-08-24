from __future__ import annotations

from .ast_normalize import ast_normalize
from .cst_format import cst_format
from .pep8_format import pep8_format
from .black_format import black_format


def chain_format(code: str) -> str:
    """Run a chain of formatters with best-effort semantics."""
    x = ast_normalize(code)
    x = cst_format(x)
    x = pep8_format(x)
    x = black_format(x)
    return x
