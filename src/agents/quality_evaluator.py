# src/agents/quality_evaluator.py
from src.agents.base_agent import BaseAgent
from typing import Dict, Any, List 

class QualityEvaluatorAgent(BaseAgent):
    """Evaluates quality of research output - provides RL reward signal"""
    
    def __init__(self):
        super().__init__(
            name="Quality Evaluator",
            role="Assessing research output quality"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate research output quality
        
        Input: {
            'summary': str,
            'papers_count': int,
            'citations': List[str],
            'query_complexity': str
        }
        Output: {
            'quality_score': float (0-1),
            'metrics': Dict,
            'reward': float,
            'agent': str
        }
        """
        summary = input_data.get('summary', '')
        papers_count = input_data.get('papers_count', 0)
        citations = input_data.get('citations', [])
        complexity = input_data.get('query_complexity', 'moderate')
        
        # Calculate individual metrics
        completeness = self._assess_completeness(summary, papers_count)
        depth = self._assess_depth(summary)
        coherence = self._assess_coherence(summary)
        citation_quality = self._assess_citations(citations, papers_count)
        
        # Calculate overall quality score
        quality_score = (
            0.3 * completeness +
            0.3 * depth +
            0.2 * coherence +
            0.2 * citation_quality
        )
        
        # Generate reward signal for RL
        reward = self._calculate_reward(quality_score, complexity)
        
        metrics = {
            'completeness': completeness,
            'depth': depth,
            'coherence': coherence,
            'citation_quality': citation_quality
        }
        
        result = {
            'quality_score': quality_score,
            'metrics': metrics,
            'reward': reward,
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _assess_completeness(self, summary: str, papers_count: int) -> float:
        """Assess if summary covers sufficient material"""
        if papers_count == 0:
            return 0.0
        
        # Check summary length
        summary_length = len(summary)
        
        if summary_length < 100:
            return 0.3
        elif summary_length < 500:
            return 0.6
        elif summary_length < 1000:
            return 0.8
        else:
            return 1.0
    
    def _assess_depth(self, summary: str) -> float:
        """Assess depth of analysis"""
        # Simple heuristic: look for analytical keywords
        depth_keywords = ['however', 'furthermore', 'specifically', 'demonstrates', 
                         'reveals', 'suggests', 'indicates', 'novel', 'significant']
        
        count = sum(1 for keyword in depth_keywords if keyword in summary.lower())
        return min(count / 5, 1.0)  # Normalize to 0-1
    
    def _assess_coherence(self, summary: str) -> float:
        """Assess logical coherence"""
        # Simple check: presence of structure
        has_intro = 'research summary' in summary.lower() or 'findings' in summary.lower()
        has_findings = 'key findings' in summary.lower() or 'results' in summary.lower()
        has_conclusion = 'based on' in summary.lower() or 'conclusion' in summary.lower()
        
        structure_score = (has_intro + has_findings + has_conclusion) / 3
        return structure_score
    
    def _assess_citations(self, citations: List[str], papers_count: int) -> float:
        """Assess citation quality"""
        if papers_count == 0:
            return 0.0
        
        citation_ratio = len(citations) / max(papers_count, 1)
        return min(citation_ratio, 1.0)
    
    def _calculate_reward(self, quality_score: float, complexity: str) -> float:
        """Calculate RL reward signal"""
        # Base reward is quality score
        reward = quality_score
        
        # Bonus for handling complex queries well
        if complexity == 'complex' and quality_score > 0.7:
            reward += 0.2
        elif complexity == 'simple' and quality_score > 0.8:
            reward += 0.1
        
        # Normalize to -1 to 1 range
        reward = (reward * 2) - 1
        
        return reward