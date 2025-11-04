"""Knowledge Graph Builder - строит граф знаний из опыта всех систем.

Автоматически извлекает patterns, relationships и insights.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class KnowledgeNode:
    """Node in knowledge graph."""
    id: str
    type: str  # 'concept', 'pattern', 'strategy', 'outcome'
    data: Dict[str, Any]
    confidence: float
    created_at: float
    updated_at: float
    access_count: int


@dataclass
class KnowledgeEdge:
    """Edge in knowledge graph."""
    source_id: str
    target_id: str
    relationship: str  # 'causes', 'improves', 'conflicts', 'requires'
    weight: float
    evidence_count: int


class KnowledgeGraphBuilder:
    """Автоматически строит и обновляет граф знаний из опыта систем."""
    
    def __init__(self, state_path: str = "log/knowledge_graph.json"):
        self.state_path = state_path
        
        # Graph structure
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[Tuple[str, str], KnowledgeEdge] = {}
        
        # Adjacency lists for fast traversal
        self.outgoing: Dict[str, Set[str]] = defaultdict(set)
        self.incoming: Dict[str, Set[str]] = defaultdict(set)
        
        # Pattern detection
        self.patterns: Dict[str, int] = defaultdict(int)  # pattern -> frequency
        
        # Insight cache
        self.insights: deque[Dict[str, Any]] = deque(maxlen=100)
        
        # Graph statistics
        self.stats: Dict[str, Any] = {}
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load graph state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore nodes
                for node_data in data.get('nodes', []):
                    node = KnowledgeNode(
                        id=node_data['id'],
                        type=node_data['type'],
                        data=node_data['data'],
                        confidence=node_data['confidence'],
                        created_at=node_data['created_at'],
                        updated_at=node_data['updated_at'],
                        access_count=node_data['access_count']
                    )
                    self.nodes[node.id] = node
                
                # Restore edges
                for edge_data in data.get('edges', []):
                    key = (edge_data['source'], edge_data['target'])
                    edge = KnowledgeEdge(
                        source_id=edge_data['source'],
                        target_id=edge_data['target'],
                        relationship=edge_data['relationship'],
                        weight=edge_data['weight'],
                        evidence_count=edge_data['evidence_count']
                    )
                    self.edges[key] = edge
                    
                    # Rebuild adjacency
                    self.outgoing[edge.source_id].add(edge.target_id)
                    self.incoming[edge.target_id].add(edge.source_id)
                
                # Restore patterns
                self.patterns = defaultdict(int, data.get('patterns', {}))
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist graph state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize nodes
                nodes_data = [
                    {
                        'id': node.id,
                        'type': node.type,
                        'data': node.data,
                        'confidence': node.confidence,
                        'created_at': node.created_at,
                        'updated_at': node.updated_at,
                        'access_count': node.access_count
                    }
                    for node in self.nodes.values()
                ]
                
                # Serialize edges
                edges_data = [
                    {
                        'source': edge.source_id,
                        'target': edge.target_id,
                        'relationship': edge.relationship,
                        'weight': edge.weight,
                        'evidence_count': edge.evidence_count
                    }
                    for edge in self.edges.values()
                ]
                
                data = {
                    'nodes': nodes_data,
                    'edges': edges_data,
                    'patterns': dict(self.patterns),
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def add_experience(
        self,
        system: str,
        action: str,
        outcome: bool,
        context: Dict[str, Any]
    ) -> None:
        """Add experience to knowledge graph."""
        async with self._lock:
            # Create or update nodes
            action_node_id = f"{system}:{action}"
            outcome_node_id = f"outcome:{'success' if outcome else 'failure'}"
            
            # Add action node
            if action_node_id not in self.nodes:
                self.nodes[action_node_id] = KnowledgeNode(
                    id=action_node_id,
                    type='action',
                    data={'system': system, 'action': action},
                    confidence=0.5,
                    created_at=time.time(),
                    updated_at=time.time(),
                    access_count=0
                )
            
            action_node = self.nodes[action_node_id]
            action_node.access_count += 1
            action_node.updated_at = time.time()
            
            # Add outcome node
            if outcome_node_id not in self.nodes:
                self.nodes[outcome_node_id] = KnowledgeNode(
                    id=outcome_node_id,
                    type='outcome',
                    data={'outcome': outcome},
                    confidence=1.0,
                    created_at=time.time(),
                    updated_at=time.time(),
                    access_count=0
                )
            
            # Add or update edge
            edge_key = (action_node_id, outcome_node_id)
            if edge_key not in self.edges:
                self.edges[edge_key] = KnowledgeEdge(
                    source_id=action_node_id,
                    target_id=outcome_node_id,
                    relationship='produces',
                    weight=0.5,
                    evidence_count=0
                )
                
                self.outgoing[action_node_id].add(outcome_node_id)
                self.incoming[outcome_node_id].add(action_node_id)
            
            edge = self.edges[edge_key]
            edge.evidence_count += 1
            
            # Update edge weight (success rate)
            total_evidence = sum(
                e.evidence_count
                for e in self.edges.values()
                if e.source_id == action_node_id
            )
            
            if total_evidence > 0:
                edge.weight = edge.evidence_count / total_evidence
            
            # Update confidence
            action_node.confidence = edge.weight
            
            # Extract patterns
            await self._extract_patterns(system, action, outcome, context)
            
            # Periodically save
            if len(self.nodes) % 10 == 0:
                await self._save_state()
    
    async def _extract_patterns(
        self,
        system: str,
        action: str,
        outcome: bool,
        context: Dict[str, Any]
    ) -> None:
        """Extract patterns from experience."""
        try:
            # Pattern: system + action + outcome
            pattern = f"{system}:{action}:{'success' if outcome else 'failure'}"
            self.patterns[pattern] += 1
            
            # Context-based patterns
            if 'intent' in context:
                intent_pattern = f"intent:{context['intent']}:{'success' if outcome else 'failure'}"
                self.patterns[intent_pattern] += 1
            
            # Multi-system patterns
            if 'involved_systems' in context:
                systems = context['involved_systems']
                if len(systems) > 1:
                    collab_pattern = f"collab:{'+'.join(sorted(systems))}:{'success' if outcome else 'failure'}"
                    self.patterns[collab_pattern] += 1
        except Exception:
            pass
    
    async def query_knowledge(
        self,
        query: str,
        query_type: str = 'similar'
    ) -> List[Dict[str, Any]]:
        """Query knowledge graph."""
        async with self._lock:
            results = []
            
            if query_type == 'similar':
                # Find nodes with similar data
                query_lower = query.lower()
                
                for node in self.nodes.values():
                    # Simple similarity
                    node_str = str(node.data).lower()
                    if query_lower in node_str:
                        results.append({
                            'id': node.id,
                            'type': node.type,
                            'data': node.data,
                            'confidence': node.confidence
                        })
            
            elif query_type == 'connected':
                # Find connected nodes
                if query in self.nodes:
                    connected = self.outgoing.get(query, set())
                    
                    for node_id in connected:
                        node = self.nodes[node_id]
                        results.append({
                            'id': node.id,
                            'type': node.type,
                            'data': node.data,
                            'confidence': node.confidence
                        })
            
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return results[:10]  # Top 10
    
    async def get_insights(self) -> List[Dict[str, Any]]:
        """Get actionable insights from knowledge graph."""
        async with self._lock:
            insights = []
            
            # Insight 1: Most successful patterns
            success_patterns = {
                pattern: count
                for pattern, count in self.patterns.items()
                if 'success' in pattern and count > 5
            }
            
            if success_patterns:
                best_pattern = max(success_patterns.items(), key=lambda x: x[1])
                insights.append({
                    'type': 'successful_pattern',
                    'pattern': best_pattern[0],
                    'frequency': best_pattern[1],
                    'recommendation': f"Pattern '{best_pattern[0]}' has high success rate"
                })
            
            # Insight 2: Failure patterns to avoid
            failure_patterns = {
                pattern: count
                for pattern, count in self.patterns.items()
                if 'failure' in pattern and count > 3
            }
            
            if failure_patterns:
                worst_pattern = max(failure_patterns.items(), key=lambda x: x[1])
                insights.append({
                    'type': 'failure_pattern',
                    'pattern': worst_pattern[0],
                    'frequency': worst_pattern[1],
                    'recommendation': f"Avoid pattern '{worst_pattern[0]}' - high failure rate"
                })
            
            # Insight 3: Underutilized nodes
            underutilized = [
                node for node in self.nodes.values()
                if node.access_count < 5 and node.confidence > 0.7
            ]
            
            if underutilized:
                insights.append({
                    'type': 'underutilized',
                    'count': len(underutilized),
                    'recommendation': f"{len(underutilized)} high-confidence actions are underutilized"
                })
            
            # Insight 4: Graph connectivity
            isolated = [
                node_id for node_id in self.nodes.keys()
                if not self.outgoing[node_id] and not self.incoming[node_id]
            ]
            
            if isolated:
                insights.append({
                    'type': 'isolated_nodes',
                    'count': len(isolated),
                    'recommendation': f"{len(isolated)} isolated nodes - need more connections"
                })
            
            return insights
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'patterns': len(self.patterns),
            'node_types': {
                node_type: len([n for n in self.nodes.values() if n.type == node_type])
                for node_type in set(n.type for n in self.nodes.values())
            },
            'avg_confidence': (
                sum(n.confidence for n in self.nodes.values()) / len(self.nodes)
                if self.nodes else 0.0
            ),
            'most_connected': max(
                [(node_id, len(self.outgoing[node_id]) + len(self.incoming[node_id]))
                 for node_id in self.nodes.keys()],
                key=lambda x: x[1],
                default=('none', 0)
            )[0]
        }


# Singleton
_knowledge_graph: Optional[KnowledgeGraphBuilder] = None
_kg_lock = asyncio.Lock()


async def get_knowledge_graph() -> KnowledgeGraphBuilder:
    """Get singleton knowledge graph."""
    global _knowledge_graph
    if _knowledge_graph is None:
        async with _kg_lock:
            if _knowledge_graph is None:
                _knowledge_graph = KnowledgeGraphBuilder()
    return _knowledge_graph


async def add_to_knowledge_graph(
    system: str,
    action: str,
    outcome: bool,
    context: Dict[str, Any]
) -> None:
    """Add experience to knowledge graph."""
    kg = await get_knowledge_graph()
    await kg.add_experience(system, action, outcome, context)


async def query_knowledge_graph(query: str, query_type: str = 'similar') -> List[Dict[str, Any]]:
    """Query knowledge graph."""
    kg = await get_knowledge_graph()
    return await kg.query_knowledge(query, query_type)


__all__ = [
    "KnowledgeGraphBuilder",
    "get_knowledge_graph",
    "add_to_knowledge_graph",
    "query_knowledge_graph",
]
