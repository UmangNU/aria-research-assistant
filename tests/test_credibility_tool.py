# tests/test_credibility_tool.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.vector_store import VectorStore
from src.agents.orchestrator import AgentOrchestrator
from src.tools.credibility_tool import CredibilityTool
import json

def test_credibility_tool():
    print("\n" + "="*70)
    print("TESTING CREDIBILITY TOOL")
    print("="*70)
    
    # Test tool directly
    print("\n1. Testing tool directly...")
    tool = CredibilityTool()
    
    sample_paper = {
        'metadata': {
            'title': 'Novel Deep Learning Approach for NeurIPS 2025',
            'published': '2025-11-01',
            'authors': 'John Doe, Jane Smith, Bob Johnson',
            'categories': ['cs.LG', 'cs.AI']
        }
    }
    
    result = tool.score_paper(sample_paper)
    print(f"\n   Tool: {tool.name}")
    print(f"   Sample Paper: {sample_paper['metadata']['title'][:50]}...")
    print(f"   Credibility Score: {result['credibility_score']}")
    print(f"   Assessment: {result['assessment']}")
    print(f"   Breakdown:")
    for metric, value in result['breakdown'].items():
        print(f"     - {metric}: {value}")
    
    # Test in full system
    print("\n" + "="*70)
    print("2. Testing in full agent system...")
    print("="*70)
    
    vs = VectorStore()
    with open('data/papers/arxiv_papers.json', 'r') as f:
        papers = json.load(f)
    vs.add_papers(papers[:100])
    
    orchestrator = AgentOrchestrator(vs)
    
    result = orchestrator.research(
        query="What are recent advances in transformer models?",
        config={'top_k': 5, 'depth': 'moderate', 'style': 'detailed'}
    )
    
    print("\n" + "="*70)
    print("3. Credibility Analysis of Retrieved Papers:")
    print("="*70)
    
    # Show credibility scores
    sources = [log[1] for log in result['execution_log'] if log[0] == 'source_discovery'][0]
    
    for i, paper in enumerate(sources['papers'][:5], 1):
        cred = paper['credibility']
        print(f"\n{i}. {paper['metadata']['title'][:60]}...")
        print(f"   Relevance: {paper['score']:.3f}")
        print(f"   Credibility: {cred['credibility_score']:.3f} ({cred['assessment']})")
        print(f"   Combined Score: {paper['combined_score']:.3f}")
    
    print("\n" + "="*70)
    print("✅ Credibility tool test complete!")
    print("="*70)

if __name__ == "__main__":
    test_credibility_tool()