# src/rag/retrieval_pipeline.py
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker
from src.rag.credibility_scorer import CredibilityScorer
from typing import List, Dict

class RetrievalPipeline:
    """Complete RAG pipeline with reranking and credibility scoring"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        self.credibility_scorer = CredibilityScorer()
    
    def retrieve(self, query: str, top_k: int = 5, domain: str = None) -> List[Dict]:
        """
        Complete retrieval pipeline:
        1. Vector search (top 50)
        2. Rerank (top 20)
        3. Credibility score
        4. Final ranking (top k)
        """
        
        # Step 1: Initial retrieval
        filter_dict = {'domain': domain} if domain else None
        candidates = self.vector_store.search(query, top_k=50, filter_dict=filter_dict)
        
        print(f"Retrieved {len(candidates)} candidates")
        
        # Step 2: Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=20)
        
        print(f"Reranked to top {len(reranked)}")
        
        # Step 3: Add credibility scores
        for doc in reranked:
            doc['credibility_score'] = self.credibility_scorer.score(doc['metadata'])
        
        # Step 4: Final ranking (combined score)
        for doc in reranked:
            # Combine retrieval, reranking, and credibility
            doc['final_score'] = (
                0.4 * doc['score'] +  # Original vector similarity
                0.4 * doc['rerank_score'] +  # Reranking score
                0.2 * doc['credibility_score']  # Credibility
            )
        
        # Sort by final score
        reranked.sort(key=lambda x: x['final_score'], reverse=True)
        
        return reranked[:top_k]