"""User Profile Learner - Personalized adaptation to user style.

Features:
- Per-user embedding centroids
- Few-shot adaptation
- User preference learning
- Style transfer
- Personalized confidence calibration
"""

from __future__ import annotations

import asyncio
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json
import os


@dataclass
class UserProfile:
    """User-specific learned profile."""
    user_id: str
    
    # Learned centroids per task (personalized)
    task_centroids: Dict[str, np.ndarray]
    
    # Preferred task types (frequency)
    task_preferences: Dict[str, float]
    
    # Average confidence (for calibration)
    avg_confidence: float
    
    # Style embedding (what makes this user unique)
    style_embedding: Optional[np.ndarray]
    
    # Training examples
    num_examples: int
    
    # Last update
    last_updated: float


class UserProfileLearner:
    """
    Learns user-specific patterns and adapts predictions.
    
    Uses few-shot learning to quickly adapt to individual users.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        
        # User profiles
        self._profiles: Dict[str, UserProfile] = {}
        
        # Training buffers per user
        self._user_examples: Dict[str, deque] = {}
        
        # Learning parameters
        self._learning_rate = 0.15  # Higher for fast adaptation
        self._min_examples_for_adaptation = 5  # Few-shot
        
        # Storage
        self._storage_dir = '.jinx/user_profiles'
    
    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        
        async with self._lock:
            if user_id not in self._profiles:
                # Try to load from disk
                profile = await self._load_profile(user_id)
                
                if profile is None:
                    # Create new profile
                    profile = UserProfile(
                        user_id=user_id,
                        task_centroids={},
                        task_preferences={},
                        avg_confidence=0.7,
                        style_embedding=None,
                        num_examples=0,
                        last_updated=time.time()
                    )
                
                self._profiles[user_id] = profile
                self._user_examples[user_id] = deque(maxlen=1000)
            
            return self._profiles[user_id]
    
    async def adapt_prediction(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        base_prediction: Tuple[str, float]
    ) -> Tuple[str, float]:
        """
        Adapt prediction for specific user.
        
        Args:
            user_id: User identifier
            query_embedding: Query embedding vector
            base_prediction: (task_type, confidence) from base model
        
        Returns:
            Adapted (task_type, confidence)
        """
        
        profile = await self.get_or_create_profile(user_id)
        
        # Not enough data yet - return base prediction
        if profile.num_examples < self._min_examples_for_adaptation:
            return base_prediction
        
        async with self._lock:
            # Check user-specific centroids
            if not profile.task_centroids:
                return base_prediction
            
            # Compute similarities to user's centroids
            user_scores = {}
            
            for task_type, centroid in profile.task_centroids.items():
                sim = self._cosine_similarity(query_embedding, centroid)
                user_scores[task_type] = sim
            
            # Best match from user profile
            if user_scores:
                user_best_task = max(user_scores.items(), key=lambda x: x[1])
                user_task, user_conf = user_best_task
                
                # Blend base prediction with user-specific
                base_task, base_conf = base_prediction
                
                # If user profile strongly suggests different task
                if user_task != base_task and user_conf > 0.7:
                    # Trust user profile more
                    adapted_conf = 0.3 * base_conf + 0.7 * user_conf
                    return (user_task, adapted_conf)
                
                # Same task - boost confidence
                elif user_task == base_task:
                    boosted_conf = min(0.95, base_conf + 0.1)
                    return (base_task, boosted_conf)
            
            # Apply preference bias
            base_task, base_conf = base_prediction
            if base_task in profile.task_preferences:
                preference_weight = profile.task_preferences[base_task]
                adapted_conf = base_conf * (1.0 + 0.2 * preference_weight)
                return (base_task, min(0.95, adapted_conf))
            
            return base_prediction
    
    async def learn_from_interaction(
        self,
        user_id: str,
        query: str,
        query_embedding: np.ndarray,
        true_task_type: str,
        outcome_quality: float
    ):
        """
        Learn from user interaction (online learning).
        
        Args:
            user_id: User identifier
            query: Query text
            query_embedding: Embedding vector
            true_task_type: Actual task type
            outcome_quality: How well it worked (0-1)
        """
        
        profile = await self.get_or_create_profile(user_id)
        
        async with self._lock:
            # Store example
            if user_id not in self._user_examples:
                self._user_examples[user_id] = deque(maxlen=1000)
            
            self._user_examples[user_id].append({
                'query': query,
                'embedding': query_embedding,
                'task_type': true_task_type,
                'quality': outcome_quality,
                'timestamp': time.time()
            })
            
            # Update centroids (online learning)
            if true_task_type not in profile.task_centroids:
                # Initialize centroid
                profile.task_centroids[true_task_type] = query_embedding.copy()
            else:
                # Move centroid towards this example
                centroid = profile.task_centroids[true_task_type]
                lr = self._learning_rate * outcome_quality
                
                profile.task_centroids[true_task_type] = (
                    (1 - lr) * centroid + lr * query_embedding
                )
            
            # Update preferences (frequency)
            profile.task_preferences[true_task_type] = (
                profile.task_preferences.get(true_task_type, 0) + 1
            )
            
            # Normalize preferences
            total = sum(profile.task_preferences.values())
            if total > 0:
                profile.task_preferences = {
                    k: v / total
                    for k, v in profile.task_preferences.items()
                }
            
            # Update style embedding (average of all examples)
            examples = list(self._user_examples[user_id])
            if len(examples) >= 10:
                embeddings = [ex['embedding'] for ex in examples[-50:]]
                profile.style_embedding = np.mean(embeddings, axis=0)
            
            # Update metadata
            profile.num_examples += 1
            profile.last_updated = time.time()
            
            # Periodic save
            if profile.num_examples % 10 == 0:
                await self._save_profile(profile)
    
    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cosine similarity."""
        
        dot = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
        
        return float(dot / (norm_x * norm_y))
    
    async def _save_profile(self, profile: UserProfile):
        """Save profile to disk."""
        
        try:
            os.makedirs(self._storage_dir, exist_ok=True)
            
            # Convert to serializable format
            data = {
                'user_id': profile.user_id,
                'task_centroids': {
                    task: centroid.tolist()
                    for task, centroid in profile.task_centroids.items()
                },
                'task_preferences': profile.task_preferences,
                'avg_confidence': profile.avg_confidence,
                'style_embedding': (
                    profile.style_embedding.tolist()
                    if profile.style_embedding is not None else None
                ),
                'num_examples': profile.num_examples,
                'last_updated': profile.last_updated
            }
            
            path = os.path.join(self._storage_dir, f"{profile.user_id}.json")
            
            with open(path, 'w') as f:
                json.dump(data, f)
        
        except Exception:
            pass
    
    async def _load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load profile from disk."""
        
        try:
            path = os.path.join(self._storage_dir, f"{user_id}.json")
            
            if not os.path.exists(path):
                return None
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            profile = UserProfile(
                user_id=data['user_id'],
                task_centroids={
                    task: np.array(centroid, dtype=np.float32)
                    for task, centroid in data['task_centroids'].items()
                },
                task_preferences=data['task_preferences'],
                avg_confidence=data['avg_confidence'],
                style_embedding=(
                    np.array(data['style_embedding'], dtype=np.float32)
                    if data['style_embedding'] is not None else None
                ),
                num_examples=data['num_examples'],
                last_updated=data['last_updated']
            )
            
            return profile
        
        except Exception:
            return None
    
    def get_profile_stats(self, user_id: str) -> Optional[Dict]:
        """Get statistics for user profile."""
        
        if user_id not in self._profiles:
            return None
        
        profile = self._profiles[user_id]
        
        return {
            'user_id': profile.user_id,
            'num_examples': profile.num_examples,
            'task_preferences': profile.task_preferences,
            'num_learned_tasks': len(profile.task_centroids),
            'has_style_embedding': profile.style_embedding is not None,
            'last_updated': profile.last_updated,
            'days_since_update': (time.time() - profile.last_updated) / 86400
        }


# Singleton
_user_learner: Optional[UserProfileLearner] = None
_learner_lock = asyncio.Lock()


async def get_user_learner() -> UserProfileLearner:
    """Get singleton user profile learner."""
    global _user_learner
    if _user_learner is None:
        async with _learner_lock:
            if _user_learner is None:
                _user_learner = UserProfileLearner()
    return _user_learner


__all__ = [
    "UserProfileLearner",
    "UserProfile",
    "get_user_learner",
]
