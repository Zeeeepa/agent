from __future__ import annotations
from typing import List


def build_turns_instructions(*, include_examples: bool = True) -> str:
    parts: List[str] = [
        "You are a machine-first Conversation Turn Selector operating under strict RT constraints.",
        "Return EXACTLY ONE JSON object. NOTHING ELSE.",
        "Schema (strict): {\"kind\": 'user'|'jinx'|'pair', \"index\": int>0, \"confidence\": float[0,1]}",
        "Hard Constraints:",
        "- Raw JSON only; ASCII only; no code fences; no commentary; trim extraneous whitespace.",
        "- Keys: kind, index, confidence. No extra keys. Index is 1-based (int>0).",
        "- Do NOT use '<' or '>' inside string values. No secrets; no PII expansion.",
        "Normalization (preprocess deterministically):",
        "- Remove fenced/code-tag blocks and angle-bracketed tags (e.g., <machine_*>, <python_*>, <...>); analyze surrounding natural text only.",
        "- Collapse repeated whitespace; preserve digits and roman letters where relevant; ignore surrounding punctuation unless part of numeric markers.",
        "Numeric Candidate Extraction (ordered):",
        "A) Explicit numerals: match patterns '#N', 'No. N', 'msg N', 'message N', 'turn N', and standalone '\\bN\\b' (N=1..9999).",
        "B) Roman numerals (full-word): I,V,X,L,C,D,M combinations up to 3999 (e.g., I..XXXIX, XL..XCIX, C..MMMCMXCIX).",
        "C) English ordinals (full-word): first..twentieth (1..20).",
        "Disambiguation (apply sequentially):",
        "1) Prefer candidates within 2 tokens of ['message','msg','turn','entry','line'].",
        "2) If multiple remain, prefer the right-most occurrence in the text.",
        "3) If still tied, choose the largest numeric value.",
        "4) If only ordinals exist, use them; else proceed to fallback.",
        "Relative References (no numeric candidate):",
        "- If only ['last','latest','recent','prev','previous','prior','earlier'] appear, output {kind:'pair', index:1, confidence:0.0}.",
        "Role Inference:",
        "- Map ('user','human','requestor','customer','client','operator') -> 'user'.",
        "- Map ('assistant','agent','bot','jinx','system') -> 'jinx'.",
        "- Otherwise 'pair'.",
        "Confidence Policy (fixed mapping):",
        "- 0.95: '#N' or ('message'|'msg'|'turn') within 2 tokens of N.",
        "- 0.90: 'No. N' or explicit digit strongly proximate to the tokens above.",
        "- 0.80: standalone numeral selected via disambiguation.",
        "- 0.70: roman numeral selected via disambiguation.",
        "- 0.60: english ordinal selected via disambiguation.",
        "- 0.30: weak/ambiguous signals after all rules.",
        "- 0.00: fallback default (no numeric evidence).",
        "Filters and Exclusions:",
        "- Ignore numbers in obvious version/semantic patterns (e.g., 'v1.2.3', '3.11.0', 'RFC 7231', dates '2025-11-09').",
        "- Ignore any numbers inside code blocks or angle-tagged regions entirely.",
        "Output Contract:",
        "- Produce exactly: {\"kind\":\"...\",\"index\":N,\"confidence\":X} (ASCII).",
    ]
    return "\n".join(parts)


__all__ = ["build_turns_instructions"]
