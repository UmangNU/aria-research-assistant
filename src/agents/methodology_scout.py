# src/agents/methodology_scout.py
from src.agents.base_agent import BaseAgent
from typing import Dict, Any, List

class MethodologyScoutAgent(BaseAgent):
    """Identifies research methodologies and approaches"""
    
    def __init__(self):
        super().__init__(
            name="Methodology Scout",
            role="Identifying research methods and experimental designs"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify methodologies
        
        Input: {
            'papers': List[Dict]
        }
        Output: {
            'methodologies': List[str],
            'common_approaches': List[str],
            'agent': str
        }
        """
        papers = input_data.get('papers', [])
        
        methodologies = self._extract_methodologies(papers)
        common_approaches = self._find_common_approaches(papers)
        
        result = {
            'methodologies': methodologies,
            'common_approaches': common_approaches,
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _extract_methodologies(self, papers: List[Dict]) -> List[str]:
        """Extract research methodologies"""
        methodologies = []
        
        method_keywords = {
            'experimental': ['experiment', 'trial', 'test', 'empirical'],
            'simulation': ['simulation', 'model', 'simulate'],
            'survey': ['survey', 'review', 'analysis'],
            'theoretical': ['theoretical', 'framework', 'formulation']
        }
        
        for paper in papers:
            title = paper.get('metadata', {}).get('title', '').lower()
            for method_type, keywords in method_keywords.items():
                if any(kw in title for kw in keywords):
                    methodologies.append(method_type)
                    break
        
        # Count occurrences
        from collections import Counter
        method_counts = Counter(methodologies)
        return [f"{method}: {count} papers" for method, count in method_counts.most_common()]
    
    def _find_common_approaches(self, papers: List[Dict]) -> List[str]:
        """Find commonly used approaches"""
        approaches = []
        
        approach_keywords = ['learning', 'optimization', 'prediction', 'classification', 
                           'detection', 'generation', 'analysis']
        
        for paper in papers:
            title = paper.get('metadata', {}).get('title', '').lower()
            for approach in approach_keywords:
                if approach in title and approach not in approaches:
                    approaches.append(approach)
        
        return approaches[:5]