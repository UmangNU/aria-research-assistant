# src/rag/vector_store.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
import json
import os

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.papers = []
        self.vectors = None
        print("✓ Vector store initialized")
    
    def add_papers(self, papers: List[Dict]):
        """Add papers to vector store"""
        print(f"Adding {len(papers)} papers...")
        
        self.papers = papers
        
        # Create documents
        documents = []
        for paper in papers:
            text = f"{paper['title']} {paper['abstract']}"
            documents.append(text)
        
        # Create TF-IDF vectors
        print("Generating embeddings...")
        self.vectors = self.vectorizer.fit_transform(documents)
        
        print(f"✓ All {len(papers)} papers indexed!")
    
    def search(self, query: str, top_k: int = 20):
        """Search for relevant papers"""
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        matches = []
        for idx in top_indices:
            paper = self.papers[idx]
            match = {
                'id': f"paper_{idx}",
                'score': float(similarities[idx]),
                'metadata': {
                    'title': paper['title'],
                    'domain': paper['domain'],
                    'published': paper['published'][:10],
                    'authors': ','.join(paper['authors'][:3])
                }
            }
            matches.append(match)
        
        return matches