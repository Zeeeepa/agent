from __future__ import annotations

from . import register_prompt


def _load() -> str:
    # Template for OpenAI cross-encoder style reranking prompt.
    # Use str.format with: query, candidate
    return (
        "You are a reranker. Rate how relevant the candidate is to the query for code/doc search. "
        "Return ONLY one ASCII floating point number between 0.0 and 1.0 — digits and optional single '.' only; no spaces, no words.\n"
        "Query: {query}\n"
        "Candidate: {candidate}\n"
    )


register_prompt("cross_rerank", _load)
