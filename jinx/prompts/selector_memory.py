from __future__ import annotations

from typing import List, Optional


def build_memory_instructions(
    *,
    allowed: Optional[List[str]] = None,
    max_k: int = 8,
    include_examples: bool = True,
) -> str:
    al = [a.strip().lower() for a in (allowed or ["turn", "memroute", "pins", "none"]) if a.strip()]
    if not al:
        al = ["turn", "memroute", "pins", "none"]
    kmax = max(1, min(16, int(max_k)))
    parts: List[str] = [
        "You are an Enterprise Memory Intent Selector operating under corporate compliance and strict RT constraints.",
        "Return EXACTLY ONE JSON object. NOTHING ELSE.",
        "Schema (strict): {\"action\": '" + "|".join(al) + "', \"params\": object, \"confidence\": float[0,1]}",
        "Params by action:",
        "- turn: {kind: 'user'|'jinx'|'pair', index: int>0}",
        (f"- memroute: {{query: string, k: int in [1,{kmax}]}}" if "memroute" in al else ""),
        ("- pins: {op: 'list'}" if "pins" in al else ""),
        "Compliance & RT:",
        "- No secrets; no PII expansion. Keep inputs sanitized. Deterministic output only.",
        "- Keep answers short and structured; avoid narrative.",
        "Guidelines:",
        "- Language-agnostic numerals: '#12', 'No. 12', '№ 12', 'N° 12', '第 １２', roman 'IV', digits '4', fullwidth '１２'.",
        "- Numbered message requests -> action 'turn'.",
        "- Recall/find/summarize requests -> action 'memroute' with concise query.",
        "- Pins management requests -> action 'pins' with appropriate op.",
        "- If unclear -> action 'none' with empty params and low confidence.",
        "Deterministic contract:",
        "- Output must be raw JSON with no code fences or commentary.",
    ]
    if include_examples:
        parts += [
            "Examples:",
            "Input: 'что я написал вторым сообщением?'",
            "Output: {\"action\":\"turn\",\"params\":{\"kind\":\"user\",\"index\":2},\"confidence\":0.9}",
            "Input: 'what did Jinx reply in message #3?'",
            "Output: {\"action\":\"turn\",\"params\":{\"kind\":\"jinx\",\"index\":3},\"confidence\":0.9}",
            "Input: 'Find facts about JWT rotation from memory'",
            f"Output: {{\"action\":\"memroute\",\"params\":{{\"query\":\"JWT rotation\",\"k\":{kmax}}},\"confidence\":0.8}}",
            "Input: 'show pins'",
            "Output: {\"action\":\"pins\",\"params\":{\"op\":\"list\"},\"confidence\":0.8}",
            "Input: 'help'",
            "Output: {\"action\":\"none\",\"params\":{},\"confidence\":0.1}",
        ]
    return "\n".join([ln for ln in parts if ln])


def build_memory_program_instructions(
    *,
    allowed: Optional[List[str]] = None,
    max_ops: int = 4,
    max_k: int = 8,
    include_examples: bool = True,
) -> str:
    al = [a.strip().lower() for a in (allowed or ["memroute","pins","append_channel","write_topic"]) if a.strip()]
    if not al:
        al = ["memroute","pins","append_channel","write_topic"]
    max_ops = max(1, min(8, int(max_ops)))
    kmax = max(1, min(16, int(max_k)))
    parts: List[str] = [
        "You are an Enterprise Memory Ops Planner operating under corporate policy and RT budgets.",
        "Return EXACTLY ONE JSON object. NOTHING ELSE.",
        "Schema (strict): {\"ops\": [op, ...]} (length <= " + str(max_ops) + ")",
        "Each op is one of the allowed actions: " + ", ".join(al) + ".",
        "Actions:",
        "- memroute: {action:'memroute', params:{query:string, k:int in [1," + str(kmax) + "]}}",
        "- pins: {action:'pins', params:{op:'list'|'add'|'remove', line?:string}}",
        "- append_channel: {action:'append_channel', params:{kind:'paths'|'symbols'|'prefs'|'decisions', lines:[string]}}",
        "- write_topic: {action:'write_topic', params:{name:string, lines:[string], mode:'append'|'replace'}}",
        "Compliance & RT:",
        "- No secrets; no PII expansion. Deterministic planning only.",
        "- Minimize ops; avoid duplication; prefer append/list over replace unless explicitly required.",
        "Planning rules:",
        "- Use provided input JSON {text:string, snapshot:string}.",
        "- Ground decisions in snapshot channels/topics and user intent.",
        "- Output must be raw JSON with no code fences or commentary.",
    ]
    if include_examples:
        parts += [
            "Examples:",
            "Input: {\"text\":\"pin this fact and search JWT\",\"snapshot\":\"...\"}",
            "Output: {\"ops\":[{\"action\":\"pins\",\"params\":{\"op\":\"add\",\"line\":\"JWT rotation policy\"}},{\"action\":\"memroute\",\"params\":{\"query\":\"JWT rotation\",\"k\":8}}]}",
            "Input: {\"text\":\"remember symbol parser rules\",\"snapshot\":\"...\"}",
            "Output: {\"ops\":[{\"action\":\"append_channel\",\"params\":{\"kind\":\"symbols\",\"lines\":[\"symbol: parser.parse() obeys PEG\"]}}]}",
        ]
    return "\n".join([ln for ln in parts if ln])


__all__ = [
    "build_memory_instructions",
    "build_memory_program_instructions",
]