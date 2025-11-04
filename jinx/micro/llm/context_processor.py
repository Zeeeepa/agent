"""Context Processor - machine-level processing of all embeddings tags.

Integrates <user>, <evidence>, <plan_mode>, and all <embeddings_*> tags
into unified semantic context for agent reasoning.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract content from a tag."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_embeddings_code(code_block: str) -> Dict[str, List[str]]:
    """Parse <embeddings_code> to extract patterns, functions, classes."""
    result = {
        'functions': [],
        'classes': [],
        'imports': [],
        'patterns': [],
        'paths': []
    }
    
    if not code_block:
        return result
    
    lines = code_block.split('\n')
    for line in lines:
        line = line.strip()
        
        # Extract path tokens
        if line.startswith('P') and '=' in line:
            # P1=path: /some/path
            result['paths'].append(line)
        
        # Extract function definitions
        elif 'def ' in line:
            result['functions'].append(line)
        
        # Extract class definitions
        elif 'class ' in line:
            result['classes'].append(line)
        
        # Extract imports
        elif line.startswith('import ') or line.startswith('from '):
            result['imports'].append(line)
        
        # Extract code patterns
        elif line and not line.startswith('#'):
            result['patterns'].append(line)
    
    return result


def parse_embeddings_refs(refs_block: str) -> Dict[str, List[str]]:
    """Parse <embeddings_refs> to extract API patterns and documentation."""
    result = {
        'api_calls': [],
        'usage_patterns': [],
        'documentation': []
    }
    
    if not refs_block:
        return result
    
    lines = refs_block.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # API calls
        if '(' in line and ')' in line:
            result['api_calls'].append(line)
        # Documentation
        elif ':' in line or '=' in line:
            result['documentation'].append(line)
        # Usage patterns
        else:
            result['usage_patterns'].append(line)
    
    return result


def parse_embeddings_graph(graph_block: str) -> Dict[str, List[str]]:
    """Parse <embeddings_graph> to extract architectural connections."""
    result = {
        'connections': [],
        'dependencies': [],
        'modules': []
    }
    
    if not graph_block:
        return result
    
    lines = graph_block.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Dependencies
        if '->' in line or 'import' in line.lower():
            result['dependencies'].append(line)
        # Connections
        elif ':' in line or '→' in line:
            result['connections'].append(line)
        # Modules
        else:
            result['modules'].append(line)
    
    return result


def parse_embeddings_memory(memory_block: str) -> Dict[str, List[str]]:
    """Parse <embeddings_memory> to extract learned patterns."""
    result = {
        'historical_patterns': [],
        'learned_facts': [],
        'decisions': []
    }
    
    if not memory_block:
        return result
    
    lines = memory_block.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Decisions
        if 'decided' in line.lower() or 'chose' in line.lower():
            result['decisions'].append(line)
        # Learned facts
        elif 'learned' in line.lower() or 'found' in line.lower():
            result['learned_facts'].append(line)
        # Historical patterns
        else:
            result['historical_patterns'].append(line)
    
    return result


def parse_embeddings_brain(brain_block: str) -> Dict[str, List[Tuple[str, float]]]:
    """Parse <embeddings_brain> to extract ML suggestions with scores."""
    result = {
        'suggestions': [],
        'optimizations': []
    }
    
    if not brain_block:
        return result
    
    lines = brain_block.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Format: - term: value (score)
        match = re.match(r'-\s*(.+?):\s*(.+?)\s*\(([0-9.]+)\)', line)
        if match:
            term, value, score = match.groups()
            try:
                score_float = float(score)
                if 'optim' in term.lower():
                    result['optimizations'].append((value.strip(), score_float))
                else:
                    result['suggestions'].append((value.strip(), score_float))
            except ValueError:
                pass
    
    return result


def parse_embeddings_meta(meta_block: str) -> Dict[str, Dict[str, str]]:
    """Parse <embeddings_meta> to extract token mappings."""
    result = {
        'paths': {},  # P# -> path
        'symbols': {},  # S# -> symbol
        'terms': {},  # T# -> term
        'frameworks': {},  # F# -> framework
        'imports': {},  # I# -> import
        'errors': {},  # E# -> error
        'claims': [],
        'weights': {}
    }
    
    if not meta_block:
        return result
    
    lines = meta_block.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse token mappings
        if '=' in line:
            parts = line.split('=', 1)
            token = parts[0].strip()
            value = parts[1].strip()
            
            if token.startswith('P'):
                result['paths'][token] = value
            elif token.startswith('S'):
                result['symbols'][token] = value
            elif token.startswith('T'):
                result['terms'][token] = value
            elif token.startswith('F'):
                result['frameworks'][token] = value
            elif token.startswith('I'):
                result['imports'][token] = value
            elif token.startswith('E'):
                result['errors'][token] = value
            elif token.startswith('C'):
                result['claims'].append(line)
            elif token.startswith('W'):
                result['weights'][token] = value
    
    return result


def process_all_context(text: str) -> Dict[str, any]:
    """Process all context tags into unified semantic structure.
    
    Returns structured data for agent reasoning.
    """
    context = {
        'user_task': None,
        'evidence': None,
        'plan_mode': None,
        'code': {},
        'refs': {},
        'graph': {},
        'memory': {},
        'brain': {},
        'meta': {}
    }
    
    # Extract all tags
    context['user_task'] = extract_tag_content(text, 'user')
    context['evidence'] = extract_tag_content(text, 'evidence')
    context['plan_mode'] = extract_tag_content(text, 'plan_mode')
    
    # Process embeddings
    code_block = extract_tag_content(text, 'embeddings_code')
    if code_block:
        context['code'] = parse_embeddings_code(code_block)
    
    refs_block = extract_tag_content(text, 'embeddings_refs')
    if refs_block:
        context['refs'] = parse_embeddings_refs(refs_block)
    
    graph_block = extract_tag_content(text, 'embeddings_graph')
    if graph_block:
        context['graph'] = parse_embeddings_graph(graph_block)
    
    memory_block = extract_tag_content(text, 'embeddings_memory')
    if memory_block:
        context['memory'] = parse_embeddings_memory(memory_block)
    
    brain_block = extract_tag_content(text, 'embeddings_brain')
    if brain_block:
        context['brain'] = parse_embeddings_brain(brain_block)
    
    meta_block = extract_tag_content(text, 'embeddings_meta')
    if meta_block:
        context['meta'] = parse_embeddings_meta(meta_block)
    
    return context


def build_agent_context_summary(context: Dict[str, any]) -> str:
    """Build concise summary for agent reasoning.
    
    Combines all parsed context into structured format for <machine_{key}> block.
    """
    lines = []
    
    if context['user_task']:
        lines.append(f"USER TASK: {context['user_task'][:200]}")
    
    if context['evidence']:
        lines.append(f"EVIDENCE: {context['evidence'][:150]}")
    
    if context['code'].get('paths'):
        lines.append(f"FILES: {len(context['code']['paths'])} paths available")
    
    if context['code'].get('functions'):
        lines.append(f"FUNCTIONS: {len(context['code']['functions'])} found")
    
    if context['code'].get('classes'):
        lines.append(f"CLASSES: {len(context['code']['classes'])} found")
    
    if context['refs'].get('api_calls'):
        lines.append(f"API PATTERNS: {len(context['refs']['api_calls'])} usage examples")
    
    if context['graph'].get('connections'):
        lines.append(f"ARCHITECTURE: {len(context['graph']['connections'])} connections mapped")
    
    if context['memory'].get('learned_facts'):
        lines.append(f"MEMORY: {len(context['memory']['learned_facts'])} learned patterns")
    
    if context['brain'].get('suggestions'):
        top_suggestion = max(context['brain']['suggestions'], key=lambda x: x[1]) if context['brain']['suggestions'] else None
        if top_suggestion:
            lines.append(f"ML SUGGESTION: {top_suggestion[0]} (confidence: {top_suggestion[1]:.2f})")
    
    return "\n".join(lines)


__all__ = [
    "extract_tag_content",
    "parse_embeddings_code",
    "parse_embeddings_refs",
    "parse_embeddings_graph",
    "parse_embeddings_memory",
    "parse_embeddings_brain",
    "parse_embeddings_meta",
    "process_all_context",
    "build_agent_context_summary",
]
