"""OpenAI service facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.llm.service`` to keep the public API stable.
"""

from __future__ import annotations

from jinx.micro.llm.service import (
    code_primer as code_primer,
    spark_openai as spark_openai,
)


__all__ = [
    "code_primer",
    "spark_openai",
]
