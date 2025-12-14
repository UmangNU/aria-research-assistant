# scripts/setup_vector_store.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.vector_store import VectorStore
import json

print("="*60)
print("ARIA Research Assistant - Vector Store Setup")
print("="*60)

print("\nLoading papers...")
with open('data/papers/arxiv_papers.json', 'r') as f:
    papers = json.load(f)

print(f"Found {len(papers)} papers")

# Use first 100 for quick testing
papers = papers[:100]
print(f"Using {len(papers)} papers for testing\n")

print("Initializing vector store...")
vs = VectorStore()

print("\nIndexing papers...")
vs.add_papers(papers)

print("\n" + "="*60)
print("Testing search...")
print("="*60)

# Test search
results = vs.search("deep reinforcement learning for robotics", top_k=5)

for i, r in enumerate(results):
    print(f"\n{i+1}. Score: {r['score']:.3f}")
    print(f"   Title: {r['metadata']['title']}")
    print(f"   Domain: {r['metadata']['domain']}")
    print(f"   Published: {r['metadata']['published']}")

print("\n" + "="*60)
print("✓ Vector store setup complete!")
print("="*60)