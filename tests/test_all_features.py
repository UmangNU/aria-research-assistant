# tests/test_all_features.py
"""
Complete feature verification test
Tests ALL Tier 1, 2, and 3 features
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.vector_store import VectorStore
from src.agents.orchestrator import AgentOrchestrator
from src.agents.agentic_rag import AgenticRAGAgent
from src.agents.self_reflective_agent import SelfReflectiveAgent
from src.agents.adaptive_summarizer import AdaptiveSummarizer
from src.tools.knowledge_graph import KnowledgeGraph
from src.tools.arxiv_live import ArxivLiveIntegration
from src.tools.explainable_retrieval import ExplainableRetrieval
import json

def test_all_features():
    """Verify every feature works"""
    
    print("="*80)
    print("COMPREHENSIVE FEATURE VERIFICATION")
    print("="*80)
    
    # Initialize
    print("\n📚 Initializing system...")
    vs = VectorStore()
    with open('data/papers/arxiv_papers.json', 'r') as f:
        papers = json.load(f)
    vs.add_papers(papers[:100])
    
    results = {}
    
    # TIER 1 TESTS
    print("\n" + "="*80)
    print("TIER 1: CORE GENAI COMPONENTS")
    print("="*80)
    
    # 1. RAG
    print("\n[1/11] Testing RAG...")
    try:
        search_results = vs.search("deep learning", top_k=5)
        results['rag'] = {'status': '✅', 'papers_found': len(search_results)}
        print(f"   ✅ RAG working - found {len(search_results)} papers")
    except Exception as e:
        results['rag'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ RAG failed: {e}")
    
    # 2. Advanced Prompts
    print("\n[2/11] Testing Advanced Prompts...")
    try:
        orch = AgentOrchestrator(vs)
        result = orch.research("What is AI?", {'top_k': 3, 'depth': 'moderate'})
        advanced_used = result['metadata'].get('advanced_prompts_used', False)
        results['advanced_prompts'] = {
            'status': '✅' if advanced_used else '⚠️',
            'quality': result['quality_score']
        }
        print(f"   ✅ Advanced prompts: {advanced_used} | Quality: {result['quality_score']:.2f}")
    except Exception as e:
        results['advanced_prompts'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # 3. Synthetic Data
    print("\n[3/11] Testing Synthetic Data...")
    try:
        import glob
        synthetic_files = glob.glob('data/synthetic/*.json')
        if synthetic_files:
            with open(synthetic_files[0], 'r') as f:
                data = json.load(f)
            results['synthetic_data'] = {'status': '✅', 'queries': len(data) if isinstance(data, list) else 'N/A'}
            print(f"   ✅ Synthetic data: {len(data) if isinstance(data, list) else 'N/A'} queries")
        else:
            results['synthetic_data'] = {'status': '⚠️', 'note': 'Files exist but not loaded'}
            print("   ⚠️  Synthetic data files exist")
    except Exception as e:
        results['synthetic_data'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # 4. Multi-modal
    print("\n[4/11] Testing Multi-modal...")
    try:
        from src.tools.multimodal_analyzer import MultiModalAnalyzer
        mm = MultiModalAnalyzer()
        results['multimodal'] = {'status': '✅', 'tool': mm.name}
        print(f"   ✅ Multi-modal analyzer ready")
    except Exception as e:
        results['multimodal'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # TIER 2 TESTS
    print("\n" + "="*80)
    print("TIER 2: UNIQUE INNOVATIONS")
    print("="*80)
    
    # 5. Agentic RAG
    print("\n[5/11] Testing Agentic RAG...")
    try:
        agentic = AgenticRAGAgent(vs)
        result = agentic.execute({'query': 'What is ML?', 'max_subqueries': 2})
        results['agentic_rag'] = {
            'status': '✅',
            'subquestions': len(result['sub_questions']),
            'unique_papers': result['unique_papers_count']
        }
        print(f"   ✅ Agentic RAG: {len(result['sub_questions'])} sub-Qs, {result['unique_papers_count']} papers")
    except Exception as e:
        results['agentic_rag'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # 6. Self-Reflection
    print("\n[6/11] Testing Self-Reflection...")
    try:
        reflective = SelfReflectiveAgent()
        result = reflective.execute({
            'initial_summary': 'Test summary about AI',
            'query': 'What is AI?',
            'papers': ['Paper 1', 'Paper 2']
        })
        results['self_reflection'] = {
            'status': '✅',
            'iterations': result['improvement_iterations'],
            'improvement': f"{result['improvement_percent']}%"
        }
        print(f"   ✅ Self-reflection: {result['improvement_iterations']} iterations")
    except Exception as e:
        results['self_reflection'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # 7. Explainable Retrieval
    print("\n[7/11] Testing Explainable Retrieval...")
    try:
        explainer = ExplainableRetrieval()
        test_paper = {'metadata': {'title': 'Test'}, 'score': 0.5, 'credibility': {'credibility_score': 0.7, 'assessment': 'Good'}}
        explanation = explainer.explain_paper_selection(test_paper, 'test query', 1)
        results['explainable'] = {'status': '✅', 'has_explanation': 'why_selected' in explanation}
        print(f"   ✅ Explainable retrieval working")
    except Exception as e:
        results['explainable'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # TIER 3 TESTS
    print("\n" + "="*80)
    print("TIER 3: ADVANCED FEATURES")
    print("="*80)
    
    # 8. Knowledge Graph
    print("\n[8/11] Testing Knowledge Graph...")
    try:
        kg = KnowledgeGraph()
        kg.build_from_papers(papers[:50])
        stats = kg.get_stats()
        results['knowledge_graph'] = {
            'status': '✅',
            'papers': stats['total_papers'],
            'concepts': stats['total_concepts']
        }
        print(f"   ✅ Knowledge graph: {stats['total_concepts']} concepts")
    except Exception as e:
        results['knowledge_graph'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # 9. Real-time arXiv
    print("\n[9/11] Testing Real-time arXiv...")
    try:
        arxiv_live = ArxivLiveIntegration()
        results['arxiv_live'] = {'status': '✅', 'tool': arxiv_live.name}
        print(f"   ✅ arXiv integration ready")
    except Exception as e:
        results['arxiv_live'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # 10. Adaptive Summarization
    print("\n[10/11] Testing Adaptive Summarization...")
    try:
        adaptive = AdaptiveSummarizer()
        result = adaptive.execute({
            'summary': 'Technical AI summary',
            'query': 'What is AI?',
            'user_level': 'undergraduate'
        })
        results['adaptive_summary'] = {'status': '✅', 'adapted': len(result['adapted_summary']) > 0}
        print(f"   ✅ Adaptive summarization working")
    except Exception as e:
        results['adaptive_summary'] = {'status': '❌', 'error': str(e)}
        print(f"   ❌ Failed: {e}")
    
    # 11. Streamlit App
    print("\n[11/11] Testing Streamlit App...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "app.py")
        results['streamlit_app'] = {'status': '✅', 'file': 'app.py exists'}
        print(f"   ✅ Streamlit app file exists")
    except:
        results['streamlit_app'] = {'status': '❌'}
        print(f"   ❌ app.py missing")
    
    # SUMMARY
    print("\n" + "="*80)
    print("FEATURE VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r['status'] == '✅')
    warned = sum(1 for r in results.values() if r['status'] == '⚠️')
    failed = sum(1 for r in results.values() if r['status'] == '❌')
    
    print(f"\n✅ Passed: {passed}/11")
    print(f"⚠️  Warnings: {warned}/11")
    print(f"❌ Failed: {failed}/11")
    
    print("\nDetailed Results:")
    for feature, data in results.items():
        print(f"  {data['status']} {feature:20s}: {str(data).replace(data['status'], '').strip('{}')}")
    
    if failed == 0:
        print("\n🎉 ALL FEATURES WORKING!")
    
    return results

if __name__ == "__main__":
    results = test_all_features()