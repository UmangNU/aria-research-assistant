# tests/test_rag.py
from src.rag.retrieval_pipeline import RetrievalPipeline

def test_rag_pipeline():
    pipeline = RetrievalPipeline()
    
    test_queries = [
        "transformer architecture for natural language processing",
        "deep reinforcement learning for robotics",
        "CRISPR gene editing applications",
        "quantum computing algorithms",
        "climate change modeling techniques"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = pipeline.retrieve(query, top_k=5)
        
        for i, doc in enumerate(results):
            print(f"\n{i+1}. {doc['metadata']['title']}")
            print(f"   Final Score: {doc['final_score']:.3f}")
            print(f"   Vector: {doc['score']:.3f} | Rerank: {doc['rerank_score']:.3f} | Credibility: {doc['credibility_score']:.3f}")
            print(f"   Domain: {doc['metadata']['domain']}")
            print(f"   Published: {doc['metadata']['published'][:10]}")

if __name__ == "__main__":
    test_rag_pipeline()