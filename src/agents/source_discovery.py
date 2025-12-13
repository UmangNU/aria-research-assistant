# src/agents/source_discovery.py
from src.agents.base_agent import BaseAgent
from src.rag.vector_store import VectorStore
from src.tools.credibility_tool import CredibilityTool
from src.tools.explainable_retrieval import ExplainableRetrieval
from typing import Dict, Any, List

class SourceDiscoveryAgent(BaseAgent):
    """
    Discovers papers with explainability
    Now shows WHY each paper was selected!
    """
    
    def __init__(self, vector_store: VectorStore):
        super().__init__(
            name="Source Discovery",
            role="Finding and ranking papers with explainable decisions"
        )
        self.vector_store = vector_store
        self.credibility_tool = CredibilityTool()
        self.explainer = ExplainableRetrieval()  # NEW!
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search with explanations
        
        Output now includes 'explanations' field!
        """
        query = input_data.get('query', '')
        keywords = input_data.get('keywords', [])
        top_k = input_data.get('top_k', 10)
        explain = input_data.get('explain', True)  # NEW parameter
        
        # Enhance query
        enhanced_query = f"{query} {' '.join(keywords)}"
        
        # Search
        papers = self.vector_store.search(enhanced_query, top_k=top_k)
        
        # Score credibility
        for paper in papers:
            credibility_result = self.credibility_tool.score_paper(paper)
            paper['credibility'] = credibility_result
        
        # Calculate combined scores
        for paper in papers:
            paper['combined_score'] = (
                0.6 * paper['score'] +
                0.4 * paper['credibility']['credibility_score']
            )
        
        # Sort by combined score
        papers.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Generate explanations
        explanations = []
        if explain:
            print(f"   💡 Generating explanations for top {min(top_k, len(papers))} papers...")
            for i, paper in enumerate(papers[:top_k], 1):
                exp = self.explainer.explain_paper_selection(paper, query, i)
                explanations.append(exp)
        
        # Calculate avg credibility
        avg_credibility = sum(p['credibility']['credibility_score'] for p in papers) / len(papers) if papers else 0
        
        result = {
            'papers': papers,
            'count': len(papers),
            'query_used': enhanced_query,
            'avg_credibility': round(avg_credibility, 3),
            'credibility_tool_used': self.credibility_tool.name,
            'explanations': explanations,  # NEW!
            'retrieval_strategy': self.explainer.explain_retrieval_strategy(
                total_papers=100,  # Could get from vector_store if it tracks
                retrieved=len(papers),
                strategy="Vector similarity + Credibility filtering"
            ),
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result