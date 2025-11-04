"""Chain Blocks - structured tag-based output for LLM chain results.

Similar to <embeddings_*> and <memory> - machine-processable structured blocks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def build_chain_plan_block(plan_data: Dict[str, Any]) -> str:
    """Build <chain_plan> block similar to embeddings_code.
    
    Format:
    <chain_plan>
    goal: primary objective
    strategy: sequential|parallel|adaptive
    subqueries:
    - Q1: first subquery
    - Q2: second subquery
    confidence: 0.85
    ml_params: k=10,timeout=800
    </chain_plan>
    """
    if not plan_data:
        return ""
    
    lines = []
    
    # Goal
    goal = plan_data.get('goal', '') or plan_data.get('note', '')
    if goal:
        lines.append(f"goal: {goal}")
    
    # Strategy
    strategy = plan_data.get('strategy', 'sequential')
    lines.append(f"strategy: {strategy}")
    
    # Subqueries
    subqueries = plan_data.get('sub_queries', [])
    if subqueries:
        lines.append("subqueries:")
        for i, sq in enumerate(subqueries, 1):
            lines.append(f"- Q{i}: {sq}")
    
    # Confidence (if from brain)
    if 'confidence' in plan_data:
        lines.append(f"confidence: {plan_data['confidence']:.2f}")
    
    # ML parameters
    ml_params = plan_data.get('ml_params', {})
    if ml_params:
        params_str = ",".join(f"{k}={v}" for k, v in ml_params.items())
        lines.append(f"ml_params: {params_str}")
    
    body = "\n".join(lines)
    return f"<chain_plan>\n{body}\n</chain_plan>" if body else ""


def build_chain_context_block(context_data: Dict[str, Any]) -> str:
    """Build <chain_context> block with retrieved context.
    
    Format:
    <chain_context>
    sources: 5
    quality: 0.82
    fragments:
    - F1: [path/to/file.py:10-20] relevant code snippet
    - F2: [another/file.py:50-60] more context
    tokens: 2450
    </chain_context>
    """
    if not context_data:
        return ""
    
    lines = []
    
    # Source count
    sources = context_data.get('source_count', 0)
    if sources:
        lines.append(f"sources: {sources}")
    
    # Quality score
    quality = context_data.get('quality', 0.0)
    if quality:
        lines.append(f"quality: {quality:.2f}")
    
    # Fragments
    fragments = context_data.get('fragments', [])
    if fragments:
        lines.append("fragments:")
        for i, frag in enumerate(fragments, 1):
            location = frag.get('location', 'unknown')
            content = frag.get('content', '')[:80]
            lines.append(f"- F{i}: [{location}] {content}")
    
    # Token count
    tokens = context_data.get('tokens', 0)
    if tokens:
        lines.append(f"tokens: {tokens}")
    
    body = "\n".join(lines)
    return f"<chain_context>\n{body}\n</chain_context>" if body else ""


def build_chain_intelligence_block(brain_data: Dict[str, Any]) -> str:
    """Build <chain_intelligence> block with brain suggestions.
    
    Format:
    <chain_intelligence>
    expanded_query: optimized version of user query
    memory_hints:
    - M1: relevant pattern from memory
    - M2: learned fact
    brain_suggestions:
    - S1: optimization hint (0.85)
    - S2: architectural suggestion (0.78)
    adaptive_params: k=15,timeout=1200,budget=10000
    </chain_intelligence>
    """
    if not brain_data:
        return ""
    
    lines = []
    
    # Expanded query
    expanded = brain_data.get('expanded_query', '')
    if expanded:
        lines.append(f"expanded_query: {expanded}")
    
    # Memory hints
    memory_hints = brain_data.get('memory_hints', [])
    if memory_hints:
        lines.append("memory_hints:")
        for i, hint in enumerate(memory_hints, 1):
            content = hint.get('content', '') if isinstance(hint, dict) else str(hint)
            lines.append(f"- M{i}: {content[:100]}")
    
    # Brain suggestions
    suggestions = brain_data.get('brain_suggestions', [])
    if suggestions:
        lines.append("brain_suggestions:")
        for i, sugg in enumerate(suggestions, 1):
            if isinstance(sugg, tuple):
                text, score = sugg
                lines.append(f"- S{i}: {text} ({score:.2f})")
            else:
                lines.append(f"- S{i}: {sugg}")
    
    # Adaptive parameters
    adaptive = brain_data.get('adaptive_params', {})
    if adaptive:
        params_str = ",".join(f"{k}={v}" for k, v in adaptive.items())
        lines.append(f"adaptive_params: {params_str}")
    
    body = "\n".join(lines)
    return f"<chain_intelligence>\n{body}\n</chain_intelligence>" if body else ""


def build_chain_meta_block(meta_data: Dict[str, Any]) -> str:
    """Build <chain_meta> block with metadata and mappings.
    
    Format:
    <chain_meta>
    Q1=subquery: find async functions
    Q2=subquery: locate error handling
    F1=fragment: path/to/file.py:10-20
    M1=memory: pattern about async usage
    S1=suggestion: use asyncio.gather
    execution_time_ms: 450
    cache_hit: false
    quality_score: 0.85
    </chain_meta>
    """
    if not meta_data:
        return ""
    
    lines = []
    
    # Token mappings
    mappings = meta_data.get('mappings', {})
    for token, value in mappings.items():
        lines.append(f"{token}={value}")
    
    # Execution metrics
    if 'execution_time_ms' in meta_data:
        lines.append(f"execution_time_ms: {meta_data['execution_time_ms']:.1f}")
    
    if 'cache_hit' in meta_data:
        lines.append(f"cache_hit: {str(meta_data['cache_hit']).lower()}")
    
    if 'quality_score' in meta_data:
        lines.append(f"quality_score: {meta_data['quality_score']:.2f}")
    
    # Model info
    if 'model' in meta_data:
        lines.append(f"model: {meta_data['model']}")
    
    if 'tokens_used' in meta_data:
        lines.append(f"tokens_used: {meta_data['tokens_used']}")
    
    body = "\n".join(lines)
    return f"<chain_meta>\n{body}\n</chain_meta>" if body else ""


def build_chain_outcome_block(outcome_data: Dict[str, Any]) -> str:
    """Build <chain_outcome> block with execution results.
    
    Format:
    <chain_outcome>
    success: true
    phase: planning
    subqueries_generated: 3
    context_gathered: 5_sources
    intelligence_applied: brain_enhanced
    latency_ms: 450
    quality: 0.85
    </chain_outcome>
    """
    if not outcome_data:
        return ""
    
    lines = []
    
    # Success status
    success = outcome_data.get('success', False)
    lines.append(f"success: {str(success).lower()}")
    
    # Phase
    phase = outcome_data.get('phase', 'unknown')
    lines.append(f"phase: {phase}")
    
    # Metrics
    if 'subqueries_generated' in outcome_data:
        lines.append(f"subqueries_generated: {outcome_data['subqueries_generated']}")
    
    if 'context_gathered' in outcome_data:
        lines.append(f"context_gathered: {outcome_data['context_gathered']}")
    
    if 'intelligence_applied' in outcome_data:
        lines.append(f"intelligence_applied: {outcome_data['intelligence_applied']}")
    
    # Performance
    if 'latency_ms' in outcome_data:
        lines.append(f"latency_ms: {outcome_data['latency_ms']:.1f}")
    
    if 'quality' in outcome_data:
        lines.append(f"quality: {outcome_data['quality']:.2f}")
    
    body = "\n".join(lines)
    return f"<chain_outcome>\n{body}\n</chain_outcome>" if body else ""


def compact_chain_blocks(
    plan: Optional[str] = None,
    context: Optional[str] = None,
    intelligence: Optional[str] = None,
    meta: Optional[str] = None,
    outcome: Optional[str] = None,
    *,
    max_chars: int = 5000
) -> str:
    """Compact all chain blocks similar to compact_context.
    
    Deduplicate, compress, and budget-allocate.
    """
    blocks = []
    
    if plan:
        blocks.append(plan)
    if intelligence:
        blocks.append(intelligence)
    if context:
        blocks.append(context)
    if meta:
        blocks.append(meta)
    if outcome:
        blocks.append(outcome)
    
    # Simple budgeting
    total_text = "\n\n".join(blocks)
    
    if len(total_text) <= max_chars:
        return total_text
    
    # Budget allocation
    plan_budget = int(max_chars * 0.25)
    intel_budget = int(max_chars * 0.30)
    context_budget = int(max_chars * 0.30)
    meta_budget = int(max_chars * 0.10)
    outcome_budget = int(max_chars * 0.05)
    
    compacted = []
    
    if plan and len(plan) > plan_budget:
        compacted.append(plan[:plan_budget] + "\n...")
    elif plan:
        compacted.append(plan)
    
    if intelligence and len(intelligence) > intel_budget:
        compacted.append(intelligence[:intel_budget] + "\n...")
    elif intelligence:
        compacted.append(intelligence)
    
    if context and len(context) > context_budget:
        compacted.append(context[:context_budget] + "\n...")
    elif context:
        compacted.append(context)
    
    if meta and len(meta) > meta_budget:
        compacted.append(meta[:meta_budget] + "\n...")
    elif meta:
        compacted.append(meta)
    
    if outcome and len(outcome) > outcome_budget:
        compacted.append(outcome[:outcome_budget] + "\n...")
    elif outcome:
        compacted.append(outcome)
    
    return "\n\n".join(compacted)


__all__ = [
    "build_chain_plan_block",
    "build_chain_context_block",
    "build_chain_intelligence_block",
    "build_chain_meta_block",
    "build_chain_outcome_block",
    "compact_chain_blocks",
]
