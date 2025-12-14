# src/agents/trend_forecaster.py
from src.agents.base_agent import BaseAgent
from typing import Dict, Any, List
from collections import Counter

class TrendForecasterAgent(BaseAgent):
    """Identifies emerging trends and research directions"""
    
    def __init__(self):
        super().__init__(
            name="Trend Forecaster",
            role="Identifying emerging research trends"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forecast trends
        
        Input: {
            'papers': List[Dict]
        }
        Output: {
            'emerging_trends': List[str],
            'hot_topics': List[str],
            'agent': str
        }
        """
        papers = input_data.get('papers', [])
        
        # Identify emerging trends
        trends = self._identify_trends(papers)
        hot_topics = self._find_hot_topics(papers)
        
        result = {
            'emerging_trends': trends,
            'hot_topics': hot_topics,
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _identify_trends(self, papers: List[Dict]) -> List[str]:
        """Identify emerging trends from recent papers"""
        trends = []
        
        # Check for recency - papers from 2025 are most recent
        recent_papers = [p for p in papers if '2025' in p.get('metadata', {}).get('published', '')]
        
        if recent_papers:
            trends.append(f"{len(recent_papers)} papers from 2025 indicate active research area")
        
        # Look for trend keywords
        trend_keywords = ['novel', 'new', 'emerging', 'future', 'next-generation']
        titles = [p.get('metadata', {}).get('title', '').lower() for p in papers]
        
        trend_count = sum(any(kw in title for kw in trend_keywords) for title in titles)
        if trend_count > len(papers) * 0.3:
            trends.append("High novelty focus - rapidly evolving field")
        
        return trends
    
    def _find_hot_topics(self, papers: List[Dict]) -> List[str]:
        """Find frequently occurring topics"""
        # Extract keywords from titles
        all_words = []
        for paper in papers:
            title = paper.get('metadata', {}).get('title', '').lower()
            words = title.split()
            # Filter meaningful words
            meaningful = [w for w in words if len(w) > 4 and w.isalpha()]
            all_words.extend(meaningful)
        
        # Count frequencies
        word_counts = Counter(all_words)
        hot_topics = [word for word, count in word_counts.most_common(5) if count > 1]
        
        return hot_topics