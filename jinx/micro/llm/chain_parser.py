"""Chain Parser - parse structured chain blocks similar to embeddings parser.

Parses <chain_*> blocks into structured data for machine processing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def extract_chain_block(text: str, block_type: str) -> Optional[str]:
    """Extract chain block content."""
    pattern = f"<{block_type}>(.*?)</{block_type}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_chain_plan_block(block: str) -> Dict[str, Any]:
    """Parse <chain_plan> block into structured data."""
    result = {
        'goal': '',
        'strategy': 'sequential',
        'subqueries': [],
        'confidence': 0.0,
        'ml_params': {}
    }
    
    if not block:
        return result
    
    lines = block.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Key-value pairs
        if ':' in line and not line.startswith('-'):
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'goal':
                result['goal'] = value
            elif key == 'strategy':
                result['strategy'] = value
            elif key == 'subqueries':
                current_section = 'subqueries'
            elif key == 'confidence':
                result['confidence'] = float(value)
            elif key == 'ml_params':
                # Parse: k=10,timeout=800
                for param in value.split(','):
                    if '=' in param:
                        k, v = param.split('=', 1)
                        result['ml_params'][k.strip()] = v.strip()
        
        # List items
        elif line.startswith('-') and current_section == 'subqueries':
            # Format: - Q1: query text
            match = re.match(r'-\s*Q\d+:\s*(.+)', line)
            if match:
                result['subqueries'].append(match.group(1))
    
    return result


def parse_chain_context_block(block: str) -> Dict[str, Any]:
    """Parse <chain_context> block into structured data."""
    result = {
        'source_count': 0,
        'quality': 0.0,
        'fragments': [],
        'tokens': 0
    }
    
    if not block:
        return result
    
    lines = block.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if ':' in line and not line.startswith('-'):
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'sources':
                result['source_count'] = int(value)
            elif key == 'quality':
                result['quality'] = float(value)
            elif key == 'fragments':
                current_section = 'fragments'
            elif key == 'tokens':
                result['tokens'] = int(value)
        
        elif line.startswith('-') and current_section == 'fragments':
            # Format: - F1: [location] content
            match = re.match(r'-\s*F\d+:\s*\[([^\]]+)\]\s*(.+)', line)
            if match:
                location, content = match.groups()
                result['fragments'].append({
                    'location': location,
                    'content': content
                })
    
    return result


def parse_chain_intelligence_block(block: str) -> Dict[str, Any]:
    """Parse <chain_intelligence> block into structured data."""
    result = {
        'expanded_query': '',
        'memory_hints': [],
        'brain_suggestions': [],
        'adaptive_params': {}
    }
    
    if not block:
        return result
    
    lines = block.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if ':' in line and not line.startswith('-'):
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'expanded_query':
                result['expanded_query'] = value
            elif key == 'memory_hints':
                current_section = 'memory_hints'
            elif key == 'brain_suggestions':
                current_section = 'brain_suggestions'
            elif key == 'adaptive_params':
                # Parse: k=15,timeout=1200
                for param in value.split(','):
                    if '=' in param:
                        k, v = param.split('=', 1)
                        result['adaptive_params'][k.strip()] = v.strip()
        
        elif line.startswith('-'):
            # Format: - M1: content or - S1: content (score)
            if current_section == 'memory_hints':
                match = re.match(r'-\s*M\d+:\s*(.+)', line)
                if match:
                    result['memory_hints'].append(match.group(1))
            elif current_section == 'brain_suggestions':
                match = re.match(r'-\s*S\d+:\s*(.+?)\s*\(([0-9.]+)\)', line)
                if match:
                    text, score = match.groups()
                    result['brain_suggestions'].append((text, float(score)))
                else:
                    match = re.match(r'-\s*S\d+:\s*(.+)', line)
                    if match:
                        result['brain_suggestions'].append((match.group(1), 0.0))
    
    return result


def parse_chain_meta_block(block: str) -> Dict[str, Any]:
    """Parse <chain_meta> block into structured data."""
    result = {
        'mappings': {},
        'execution_time_ms': 0.0,
        'cache_hit': False,
        'quality_score': 0.0,
        'model': '',
        'tokens_used': 0
    }
    
    if not block:
        return result
    
    lines = block.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'execution_time_ms':
                result['execution_time_ms'] = float(value)
            elif key == 'cache_hit':
                result['cache_hit'] = value.lower() == 'true'
            elif key == 'quality_score':
                result['quality_score'] = float(value)
            elif key == 'model':
                result['model'] = value
            elif key == 'tokens_used':
                result['tokens_used'] = int(value)
        
        elif '=' in line:
            # Token mappings: Q1=subquery: text
            token, mapping = line.split('=', 1)
            result['mappings'][token.strip()] = mapping.strip()
    
    return result


def parse_chain_outcome_block(block: str) -> Dict[str, Any]:
    """Parse <chain_outcome> block into structured data."""
    result = {
        'success': False,
        'phase': '',
        'subqueries_generated': 0,
        'context_gathered': '',
        'intelligence_applied': '',
        'latency_ms': 0.0,
        'quality': 0.0
    }
    
    if not block:
        return result
    
    lines = block.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
        
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        if key == 'success':
            result['success'] = value.lower() == 'true'
        elif key == 'phase':
            result['phase'] = value
        elif key == 'subqueries_generated':
            result['subqueries_generated'] = int(value)
        elif key == 'context_gathered':
            result['context_gathered'] = value
        elif key == 'intelligence_applied':
            result['intelligence_applied'] = value
        elif key == 'latency_ms':
            result['latency_ms'] = float(value)
        elif key == 'quality':
            result['quality'] = float(value)
    
    return result


def parse_all_chain_blocks(text: str) -> Dict[str, Dict[str, Any]]:
    """Parse all chain blocks from text similar to process_all_context."""
    result = {
        'plan': {},
        'context': {},
        'intelligence': {},
        'meta': {},
        'outcome': {}
    }
    
    # Extract and parse each block type
    plan_block = extract_chain_block(text, 'chain_plan')
    if plan_block:
        result['plan'] = parse_chain_plan_block(plan_block)
    
    context_block = extract_chain_block(text, 'chain_context')
    if context_block:
        result['context'] = parse_chain_context_block(context_block)
    
    intel_block = extract_chain_block(text, 'chain_intelligence')
    if intel_block:
        result['intelligence'] = parse_chain_intelligence_block(intel_block)
    
    meta_block = extract_chain_block(text, 'chain_meta')
    if meta_block:
        result['meta'] = parse_chain_meta_block(meta_block)
    
    outcome_block = extract_chain_block(text, 'chain_outcome')
    if outcome_block:
        result['outcome'] = parse_chain_outcome_block(outcome_block)
    
    return result


def build_chain_summary(parsed: Dict[str, Dict[str, Any]]) -> str:
    """Build concise summary of parsed chain data for agents."""
    lines = []
    
    plan = parsed.get('plan', {})
    if plan.get('goal'):
        lines.append(f"GOAL: {plan['goal']}")
    if plan.get('subqueries'):
        lines.append(f"SUBQUERIES: {len(plan['subqueries'])} generated")
    
    context = parsed.get('context', {})
    if context.get('source_count'):
        lines.append(f"CONTEXT: {context['source_count']} sources, quality={context.get('quality', 0):.2f}")
    
    intelligence = parsed.get('intelligence', {})
    if intelligence.get('expanded_query'):
        lines.append(f"ENHANCED: query expanded")
    if intelligence.get('brain_suggestions'):
        lines.append(f"SUGGESTIONS: {len(intelligence['brain_suggestions'])} from brain")
    
    outcome = parsed.get('outcome', {})
    if outcome.get('success'):
        lines.append(f"SUCCESS: {outcome.get('phase', 'unknown')} completed in {outcome.get('latency_ms', 0):.0f}ms")
    
    return "\n".join(lines)


__all__ = [
    "extract_chain_block",
    "parse_chain_plan_block",
    "parse_chain_context_block",
    "parse_chain_intelligence_block",
    "parse_chain_meta_block",
    "parse_chain_outcome_block",
    "parse_all_chain_blocks",
    "build_chain_summary",
]
