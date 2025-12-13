# src/agents/citation_analyzer.py
from src.agents.base_agent import BaseAgent
from typing import Dict, Any, List

class CitationAnalyzerAgent(BaseAgent):
    """Analyzes citation patterns and paper relationships"""
    
    def __init__(self):
        super().__init__(
            name="Citation Analyzer",
            role="Analyzing citation patterns and paper relationships"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze citation patterns
        
        Input: {
            'papers': List[Dict]
        }
        Output: {
            'high_impact_papers': List[Dict],
            'citation_clusters': List[str],
            'agent': str
        }
        """
        papers = input_data.get('papers', [])
        
        # Identify high-impact papers (high relevance scores)
        high_impact = sorted(papers, key=lambda x: x.get('score', 0), reverse=True)[:5]
        
        # Group papers by domain (simple clustering)
        clusters = self._cluster_by_domain(papers)
        
        result = {
            'high_impact_papers': high_impact,
            'citation_clusters': clusters,
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _cluster_by_domain(self, papers: List[Dict]) -> List[str]:
        """Cluster papers by domain"""
        from collections import Counter
        
        domains = [p.get('metadata', {}).get('domain', 'unknown') for p in papers]
        domain_counts = Counter(domains)
        
        clusters = [f"{domain}: {count} papers" for domain, count in domain_counts.most_common()]
        return clusters