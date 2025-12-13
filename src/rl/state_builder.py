# src/rl/state_builder.py
"""
State Builder for RL System
Converts agent system state into feature vectors for RL
"""

import numpy as np
from typing import Dict, Any, List
from collections import defaultdict

class StateBuilder:
    """Builds state representations for RL from agent system"""
    
    def __init__(self):
        self.domain_map = {
            'cs_ml': 0, 'cs_nlp': 1, 'cs_cv': 2,
            'biology': 3, 'physics': 4, 'medicine': 5,
            'general': 6
        }
        
        self.complexity_map = {
            'simple': 0.33,
            'moderate': 0.66,
            'complex': 1.0
        }
        
        self.query_type_map = {
            'literature_review': 0, 'comparison': 1,
            'methodology': 2, 'definition': 3,
            'exploratory': 4
        }
        
        # Track history for temporal features
        self.history = defaultdict(list)
        self.max_history = 10
    
    def build_state(self, 
                    query_analysis: Dict[str, Any],
                    sources: Dict[str, Any],
                    papers_analyzed: int,
                    current_quality: float = 0.0) -> np.ndarray:
        """
        Build state vector from current system state
        
        Args:
            query_analysis: Output from QueryAnalyzerAgent
            sources: Output from SourceDiscoveryAgent
            papers_analyzed: Number of papers analyzed
            current_quality: Current quality score
            
        Returns:
            State vector (1D numpy array)
        """
        
        # Query features (7 dims)
        domain = query_analysis.get('domain', 'general')
        domain_encoded = self._one_hot_encode(domain, self.domain_map, 7)
        
        complexity = self.complexity_map.get(
            query_analysis.get('complexity', 'moderate'), 0.66
        )
        
        query_type = query_analysis.get('query_type', 'exploratory')
        query_type_encoded = self._one_hot_encode(
            query_type, self.query_type_map, 5
        )
        
        # Source features (4 dims)
        papers_available = min(sources.get('count', 0) / 20.0, 1.0)  # Normalize to 0-1
        avg_credibility = sources.get('avg_credibility', 0.5)
        avg_relevance = np.mean([p['score'] for p in sources.get('papers', [])] or [0.0])
        papers_analyzed_norm = min(papers_analyzed / 10.0, 1.0)
        
        # Performance features (2 dims)
        current_quality_norm = current_quality
        
        # Historical features (3 dims)
        recent_qualities = self.history['quality'][-5:]  # Last 5 queries
        avg_recent_quality = np.mean(recent_qualities) if recent_qualities else 0.5
        quality_trend = self._calculate_trend(recent_qualities)
        query_count_norm = min(len(self.history['quality']) / 100.0, 1.0)
        
        # Concatenate all features
        state = np.concatenate([
            domain_encoded,           # 7 dims
            [complexity],             # 1 dim
            query_type_encoded,       # 5 dims
            [papers_available],       # 1 dim
            [avg_credibility],        # 1 dim
            [avg_relevance],          # 1 dim
            [papers_analyzed_norm],   # 1 dim
            [current_quality_norm],   # 1 dim
            [avg_recent_quality],     # 1 dim
            [quality_trend],          # 1 dim
            [query_count_norm]        # 1 dim
        ])
        
        # State vector: 21 dimensions total
        return state.astype(np.float32)
    
    def update_history(self, quality: float, reward: float):
        """Update history with new results"""
        self.history['quality'].append(quality)
        self.history['reward'].append(reward)
        
        # Keep only recent history
        if len(self.history['quality']) > self.max_history:
            self.history['quality'] = self.history['quality'][-self.max_history:]
            self.history['reward'] = self.history['reward'][-self.max_history:]
    
    def _one_hot_encode(self, value: str, mapping: Dict[str, int], size: int) -> np.ndarray:
        """One-hot encode a categorical value"""
        vec = np.zeros(size)
        if value in mapping:
            vec[mapping[value]] = 1.0
        return vec
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in recent values (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Fit line
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return np.clip(slope, -1.0, 1.0)
        return 0.0
    
    def get_state_dim(self) -> int:
        """Return state dimension"""
        return 21
    
    def reset_history(self):
        """Reset history"""
        self.history = defaultdict(list)