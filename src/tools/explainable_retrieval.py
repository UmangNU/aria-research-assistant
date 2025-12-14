# src/tools/explainable_retrieval.py
"""
Explainable Retrieval Tool
Provides transparency into WHY papers were selected

Builds trust through explainability!
"""

from typing import Dict, Any, List
import numpy as np

class ExplainableRetrieval:
    """Explains retrieval decisions to build user trust"""
    
    def __init__(self):
        self.name = "Explainable Retrieval"
        
    def explain_paper_selection(self, 
                                paper: Dict[str, Any],
                                query: str,
                                rank: int) -> Dict[str, Any]:
        """
        Explain why this paper was selected
        
        Args:
            paper: Paper dictionary with scores
            query: Original query
            rank: Paper's rank in results (1-indexed)
        
        Returns:
            Explanation dictionary
        """
        
        metadata = paper.get('metadata', {})
        
        # Relevance explanation
        relevance_score = paper.get('score', 0)
        relevance_explanation = self._explain_relevance(metadata, query, relevance_score)
        
        # Credibility explanation
        credibility = paper.get('credibility', {})
        credibility_explanation = self._explain_credibility(credibility)
        
        # Ranking explanation
        combined_score = paper.get('combined_score', relevance_score)
        ranking_explanation = self._explain_ranking(rank, combined_score, relevance_score, credibility.get('credibility_score', 0))
        
        # Overall explanation
        explanation = {
            'paper_title': metadata.get('title', 'Unknown'),
            'rank': rank,
            'overall_score': round(combined_score, 3),
            'why_selected': self._generate_why_selected(relevance_explanation, credibility_explanation, ranking_explanation),
            'relevance': {
                'score': round(relevance_score, 3),
                'explanation': relevance_explanation
            },
            'credibility': {
                'score': round(credibility.get('credibility_score', 0), 3),
                'assessment': credibility.get('assessment', 'Unknown'),
                'breakdown': credibility.get('breakdown', {}),
                'explanation': credibility_explanation
            },
            'ranking': ranking_explanation,
            'key_factors': self._identify_key_factors(paper, query)
        }
        
        return explanation
    
    def _explain_relevance(self, metadata: Dict, query: str, score: float) -> str:
        """Explain relevance score"""
        
        title = metadata.get('title', '').lower()
        query_lower = query.lower()
        
        # Find matching keywords
        query_words = set(query_lower.split())
        title_words = set(title.split())
        matches = query_words & title_words
        
        if score > 0.3:
            explanation = f"High relevance (score: {score:.2f}). "
        elif score > 0.15:
            explanation = f"Moderate relevance (score: {score:.2f}). "
        else:
            explanation = f"Low relevance (score: {score:.2f}). "
        
        if matches:
            explanation += f"Title contains query keywords: {', '.join(list(matches)[:3])}. "
        else:
            explanation += "Semantic similarity detected through vector matching. "
        
        return explanation
    
    def _explain_credibility(self, credibility: Dict) -> str:
        """Explain credibility assessment"""
        
        if not credibility:
            return "Credibility not assessed."
        
        score = credibility.get('credibility_score', 0)
        assessment = credibility.get('assessment', 'Unknown')
        breakdown = credibility.get('breakdown', {})
        
        explanation = f"{assessment} (score: {score:.2f}). "
        
        # Identify top factors
        if breakdown:
            sorted_factors = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
            top_factor = sorted_factors[0]
            
            factor_explanations = {
                'venue_quality': f"Strong publication venue (score: {top_factor[1]:.2f})",
                'recency': f"Recent publication (score: {top_factor[1]:.2f})",
                'category_relevance': f"Relevant research category (score: {top_factor[1]:.2f})",
                'author_count': f"Collaborative work with {breakdown.get('author_count', 0):.0f} authors",
                'title_quality': f"Well-structured title (score: {top_factor[1]:.2f})"
            }
            
            explanation += factor_explanations.get(top_factor[0], "")
        
        return explanation
    
    def _explain_ranking(self, rank: int, combined: float, relevance: float, credibility: float) -> str:
        """Explain why paper ranked at this position"""
        
        if rank == 1:
            return f"Ranked #1: Best combination of relevance ({relevance:.2f}) and credibility ({credibility:.2f})."
        elif rank <= 3:
            return f"Ranked #{rank} (Top 3): High combined score ({combined:.2f}) balancing relevance and credibility."
        elif rank <= 5:
            return f"Ranked #{rank} (Top 5): Good overall score ({combined:.2f})."
        else:
            return f"Ranked #{rank}: Lower combined score ({combined:.2f}) but still relevant."
    
    def _generate_why_selected(self, relevance_exp: str, credibility_exp: str, ranking_exp: str) -> str:
        """Generate overall 'why selected' explanation"""
        
        return f"{ranking_exp} {relevance_exp} {credibility_exp}".strip()
    
    def _identify_key_factors(self, paper: Dict, query: str) -> List[str]:
        """Identify key factors that led to selection"""
        
        factors = []
        
        # Relevance factors
        if paper.get('score', 0) > 0.2:
            factors.append("High semantic similarity to query")
        
        # Credibility factors
        cred = paper.get('credibility', {})
        if cred.get('credibility_score', 0) > 0.7:
            factors.append("High credibility rating")
        
        breakdown = cred.get('breakdown', {})
        if breakdown.get('recency', 0) > 0.8:
            factors.append("Recent publication (2024-2025)")
        
        if breakdown.get('venue_quality', 0) > 0.8:
            factors.append("Top-tier venue")
        
        # Domain match
        query_lower = query.lower()
        domain_keywords = {
            'cs_ml': ['learning', 'neural', 'deep', 'ai', 'machine'],
            'biology': ['protein', 'gene', 'cell', 'molecular', 'crispr'],
            'physics': ['quantum', 'particle', 'energy']
        }
        
        paper_domain = paper.get('metadata', {}).get('domain', '')
        if paper_domain in domain_keywords:
            if any(kw in query_lower for kw in domain_keywords[paper_domain]):
                factors.append(f"Domain match ({paper_domain})")
        
        return factors if factors else ["Selected through vector similarity"]
    
    def explain_retrieval_strategy(self, 
                                   total_papers: int,
                                   retrieved: int,
                                   strategy: str) -> str:
        """
        Explain the overall retrieval strategy
        
        Args:
            total_papers: Total papers in database
            retrieved: Number retrieved
            strategy: Strategy used
        
        Returns:
            Strategy explanation
        """
        
        explanation = f"""**Retrieval Strategy Explanation:**

**Database:** Searched {total_papers} papers in vector store

**Strategy:** {strategy}

**Retrieved:** Top {retrieved} most relevant papers

**Selection Process:**
1. Vector similarity search (TF-IDF embeddings)
2. Credibility scoring (multi-factor assessment)
3. Combined ranking (60% relevance + 40% credibility)
4. Top-k selection

**Why this strategy?** Balances finding relevant papers (vector search) with ensuring quality (credibility filtering). This two-stage approach prevents retrieving low-quality but keyword-matched papers.
"""
        
        return explanation