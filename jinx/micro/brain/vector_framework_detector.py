"""Vector-Based Framework Detector - Auto-discovery, NO hardcoded lists.

Replaces: brain/scanners/frameworks.py

Features:
- Cluster-based framework detection
- Auto-discovery (no hardcoded framework names)
- Learns from embeddings
- Language-agnostic
"""

from __future__ import annotations

import asyncio
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FrameworkCluster:
    """Discovered framework cluster."""
    cluster_id: int
    centroid: np.ndarray
    examples: List[str]  # Import/package names
    confidence: float
    usage_count: int


class VectorFrameworkDetector:
    """
    Auto-discovering framework detector using clustering.
    
    NO hardcoded framework lists - discovers patterns automatically.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # Discovered framework clusters
        self._clusters: List[FrameworkCluster] = []
        
        # Cache of seen imports
        self._import_cache: Dict[str, np.ndarray] = {}
        
        # Maximum clusters
        self._max_clusters = 20
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize with seed clusters."""
        
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            # Will build clusters from actual usage
            self._initialized = True
    
    async def detect_frameworks(
        self,
        imports: List[str],
        project_root: str
    ) -> Set[str]:
        """
        Detect frameworks from imports using clustering.
        
        Args:
            imports: List of import names
            project_root: Project root path
        
        Returns:
            Set of detected framework cluster IDs (as strings)
        """
        
        if not self._initialized:
            await self.initialize()
        
        if not imports:
            return set()
        
        async with self._lock:
            detected = set()
            
            for import_name in imports[:128]:  # Limit
                try:
                    # Get embedding for import
                    import_vec = await self._get_import_embedding(import_name)
                    
                    if import_vec is None:
                        continue
                    
                    # Find nearest cluster
                    cluster_id, distance = self._find_nearest_cluster(import_vec)
                    
                    if distance < 0.5:  # Similar to existing cluster
                        detected.add(f"cluster_{cluster_id}")
                        
                        # Update cluster
                        if cluster_id < len(self._clusters):
                            self._clusters[cluster_id].usage_count += 1
                            self._clusters[cluster_id].examples.append(import_name)
                    
                    else:
                        # Create new cluster if we haven't hit limit
                        if len(self._clusters) < self._max_clusters:
                            new_cluster = FrameworkCluster(
                                cluster_id=len(self._clusters),
                                centroid=import_vec,
                                examples=[import_name],
                                confidence=0.5,
                                usage_count=1
                            )
                            self._clusters.append(new_cluster)
                            detected.add(f"cluster_{new_cluster.cluster_id}")
                
                except Exception:
                    continue
            
            return detected
    
    async def _get_import_embedding(self, import_name: str) -> Optional[np.ndarray]:
        """Get embedding for import name."""
        
        # Check cache
        if import_name in self._import_cache:
            return self._import_cache[import_name]
        
        try:
            from jinx.micro.embeddings.pipeline import embed_text
            
            # Create semantic description
            text = f"python package module {import_name} import library"
            
            emb_obj = await embed_text(text, source='framework_detector')
            
            if emb_obj and hasattr(emb_obj, 'embedding'):
                vec = np.array(emb_obj.embedding, dtype=np.float32)
                self._import_cache[import_name] = vec
                return vec
        
        except Exception:
            pass
        
        return None
    
    def _find_nearest_cluster(self, vec: np.ndarray) -> Tuple[int, float]:
        """
        Find nearest cluster to vector.
        
        Returns:
            (cluster_id, distance)
        """
        
        if not self._clusters:
            return (0, 1.0)  # No clusters yet
        
        min_dist = float('inf')
        nearest_id = 0
        
        for cluster in self._clusters:
            dist = self._euclidean_distance(vec, cluster.centroid)
            
            if dist < min_dist:
                min_dist = dist
                nearest_id = cluster.cluster_id
        
        return (nearest_id, min_dist)
    
    def _euclidean_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Euclidean distance between vectors."""
        
        # Handle dimension mismatch
        if len(x) != len(y):
            min_dim = min(len(x), len(y))
            x = x[:min_dim]
            y = y[:min_dim]
        
        return float(np.linalg.norm(x - y))
    
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict]:
        """Get information about a cluster."""
        
        if cluster_id >= len(self._clusters):
            return None
        
        cluster = self._clusters[cluster_id]
        
        # Find most common examples (representative)
        from collections import Counter
        example_counts = Counter(cluster.examples)
        top_examples = [name for name, _ in example_counts.most_common(5)]
        
        return {
            'cluster_id': cluster.cluster_id,
            'usage_count': cluster.usage_count,
            'confidence': cluster.confidence,
            'representative_imports': top_examples,
            'total_examples': len(cluster.examples)
        }
    
    def get_all_clusters(self) -> List[Dict]:
        """Get info for all clusters."""
        
        return [
            self.get_cluster_info(i)
            for i in range(len(self._clusters))
        ]


# Singleton
_framework_detector: Optional[VectorFrameworkDetector] = None
_detector_lock = asyncio.Lock()


async def get_framework_detector() -> VectorFrameworkDetector:
    """Get singleton framework detector."""
    global _framework_detector
    if _framework_detector is None:
        async with _detector_lock:
            if _framework_detector is None:
                _framework_detector = VectorFrameworkDetector()
                await _framework_detector.initialize()
    return _framework_detector


__all__ = [
    "VectorFrameworkDetector",
    "FrameworkCluster",
    "get_framework_detector",
]
