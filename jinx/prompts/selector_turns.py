from __future__ import annotations

from typing import List


def build_turns_instructions(*, include_examples: bool = True) -> str:
    parts: List[str] = [
        "You are an Enterprise Conversation Turn Selector operating under corporate policy and strict RT constraints.",
        "Return EXACTLY ONE JSON object. NOTHING ELSE.",
        "Schema (strict): {\"kind\": 'user'|'jinx'|'pair', \"index\": int>0, \"confidence\": float[0,1]}",
        "Compliance & RT:",
        "- No secrets; no PII expansion. Deterministic output only.",
        "- Keep answers minimal and structured; avoid narrative.",
        "Guidelines:",
        "- Language-agnostic numerals: '#12', 'No. 12', '№ 12', 'N° 12', '第 １２', roman 'IV', digits '4', fullwidth '１２'.",
        "- Role inference: ('user','human','operator')->'user'; ('assistant','agent','bot','jinx')->'jinx'; else 'pair'.",
        "- Relative references (e.g., 'last', 'latest', 'recent', 'prev', 'previous', 'предыдущ', 'последн'): if the numeric index cannot be inferred from the text alone, default {kind:'pair', index:1, confidence:0.0}.",
        "- If multiple numbers appear, choose the most plausible turn index referenced by the text.",
        "Deterministic contract:",
        "- Output must be raw JSON with no code fences or commentary.",
    ]
    if include_examples:
        parts += [
            "Examples:",
            "Input: 'что я написал вторым сообщением?'",
            "Output: {\"kind\":\"user\",\"index\":2,\"confidence\":0.9}",
            "Input: 'what did Jinx reply in message #3?'",
            "Output: {\"kind\":\"jinx\",\"index\":3,\"confidence\":0.9}",
            "Input: '第 ４ メッセージ について'",
            "Output: {\"kind\":\"pair\",\"index\":4,\"confidence\":0.8}",
            "Input: 'Respond to No. 10'",
            "Output: {\"kind\":\"pair\",\"index\":10,\"confidence\":0.7}",
            "Input: 'IV message content?'",
            "Output: {\"kind\":\"pair\",\"index\":4,\"confidence\":0.7}",
            "Input: 'remind me'",
            "Output: {\"kind\":\"pair\",\"index\":1,\"confidence\":0.0}",
        ]
    return "\n".join(parts)


__all__ = ["build_turns_instructions"]
