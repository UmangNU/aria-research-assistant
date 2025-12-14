# src/agents/contradiction_detector.py
from src.agents.base_agent import BaseAgent
from typing import Dict, Any, List

class ContradictionDetectorAgent(BaseAgent):
    """Detects contradictions and conflicts in research findings"""
    
    def __init__(self):
        super().__init__(
            name="Contradiction Detector",
            role="Identifying conflicting findings in research"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect contradictions
        
        Input: {
            'analyzed_papers': List[Dict]
        }
        Output: {
            'contradictions': List[str],
            'controversial_topics': List[str],
            'agent': str
        }
        """
        papers = input_data.get('analyzed_papers', [])
        
        # Simple contradiction detection (keyword-based)
        contradictions = self._find_contradictions(papers)
        controversial = self._identify_controversial_topics(papers)
        
        result = {
            'contradictions': contradictions,
            'controversial_topics': controversial,
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _find_contradictions(self, papers: List[Dict]) -> List[str]:
        """Find potential contradictions"""
        contradictions = []
        
        # Look for opposing keywords in titles
        positive_terms = ['effective', 'improved', 'better', 'successful']
        negative_terms = ['failed', 'limited', 'ineffective', 'poor']
        
        titles = [p.get('title', '').lower() for p in papers]
        
        has_positive = any(any(term in title for term in positive_terms) for title in titles)
        has_negative = any(any(term in title for term in negative_terms) for title in titles)
        
        if has_positive and has_negative:
            contradictions.append("Mixed results found in literature")
        
        return contradictions
    
    def _identify_controversial_topics(self, papers: List[Dict]) -> List[str]:
        """Identify controversial research areas"""
        controversial = []
        
        titles = [p.get('title', '').lower() for p in papers]
        
        # Look for debate indicators
        debate_keywords = ['debate', 'controversy', 'challenge', 'limitation']
        
        for title in titles:
            if any(keyword in title for keyword in debate_keywords):
                controversial.append(title[:100])  # Truncate
        
        return controversial[:5]  # Top 5