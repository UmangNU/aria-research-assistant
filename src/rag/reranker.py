# src/rag/reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Dict

class Reranker:
    def __init__(self):
        # Use cross-encoder for reranking
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 10):
        """Rerank documents using cross-encoder"""
        
        # Prepare pairs for cross-encoder
        pairs = []
        for doc in documents:
            text = f"{doc['metadata']['title']} {doc['metadata']['abstract']}"
            pairs.append([query, text])
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Combine scores with documents
        scored_docs = [(score, doc) for score, doc in zip(scores, documents)]
        
        # Sort by score
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Return top k
        return [
            {'rerank_score': score, **doc} 
            for score, doc in scored_docs[:top_k]
        ]