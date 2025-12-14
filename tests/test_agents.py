# tests/test_agents.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.vector_store import VectorStore
from src.agents.orchestrator import AgentOrchestrator
import json

def test_agent_system():
    print("\n" + "="*70)
    print("TESTING MULTI-AGENT RESEARCH SYSTEM")
    print("="*70)
    
    # Load vector store
    print("\n📚 Loading vector store...")
    vs = VectorStore()
    
    # Load papers
    with open('data/papers/arxiv_papers.json', 'r') as f:
        papers = json.load(f)
    
    # Index papers
    vs.add_papers(papers[:100])
    
    # Create orchestrator
    print("\n🤖 Initializing agent orchestrator...")
    orchestrator = AgentOrchestrator(vs)
    
    # Test queries
    test_queries = [
        "What are recent advances in deep reinforcement learning?",
        "How does protein folding prediction work?",
        "What are the latest developments in transformer architectures?"
    ]
    
    results = []
    
    for query in test_queries[:1]:  # Test with first query
        print("\n" + "="*70)
        result = orchestrator.research(
            query=query,
            config={
                'top_k': 5,
                'depth': 'moderate',
                'style': 'detailed'
            }
        )
        results.append(result)
        
        # Print summary
        print("\n📊 RESULTS:")
        print(f"Query: {result['query']}")
        print(f"Quality Score: {result['quality_score']:.2f}")
        print(f"Papers Found: {result['papers_found']}")
        print(f"Papers Analyzed: {result['papers_analyzed']}")
        print(f"\n📝 Summary Preview:")
        print(result['summary'][:500] + "...\n")
    
    print("="*70)
    print("✅ Agent system test complete!")
    print("="*70)
    
    return results

if __name__ == "__main__":
    test_agent_system()