# tests/test_production_system.py
"""
Comprehensive test of production-grade ARIA system
Tests error handling, monitoring, and all features
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.vector_store import VectorStore
from src.agents.orchestrator import AgentOrchestrator
from src.utils.logger import get_logger
from src.utils.metrics_tracker import get_performance_tracker
import json

def test_production_system():
    """Test all production features"""
    
    print("="*80)
    print("TESTING PRODUCTION-GRADE ARIA SYSTEM")
    print("="*80)
    
    # Initialize
    print("\n📚 Initializing system...")
    vs = VectorStore()
    
    # Load papers
    with open('data/papers/arxiv_papers.json', 'r') as f:
        papers = json.load(f)
    
    vs.add_papers(papers[:100])  # Quick test with 100 papers
    
    # Create orchestrator
    print("🤖 Creating orchestrator with monitoring...")
    orchestrator = AgentOrchestrator(vs)
    
    # Test queries with different scenarios
    test_scenarios = [
        {
            'name': 'Normal query',
            'query': 'What are recent advances in deep learning?',
            'config': {'top_k': 5, 'depth': 'moderate', 'style': 'detailed'}
        },
        {
            'name': 'Short query',
            'query': 'AI',
            'config': {'top_k': 3, 'depth': 'shallow', 'style': 'concise'}
        },
        {
            'name': 'Complex query',
            'query': 'How do transformer architectures leverage self-attention mechanisms for natural language understanding tasks?',
            'config': {'top_k': 10, 'depth': 'deep', 'style': 'technical'}
        },
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_scenarios)}: {scenario['name']}")
        print(f"{'='*80}")
        
        try:
            result = orchestrator.research(
                query=scenario['query'],
                config=scenario['config']
            )
            
            results.append({
                'scenario': scenario['name'],
                'success': True,
                'quality': result['quality_score'],
                'reward': result['reward'],
                'papers_analyzed': result['papers_analyzed'],
                'fallback_used': result['metadata']['fallback_used']
            })
            
            # Print result summary
            print(f"\n✅ Test {i} PASSED")
            print(f"   Query: {scenario['query'][:60]}...")
            print(f"   Quality: {result['quality_score']:.3f}")
            print(f"   Reward: {result['reward']:.3f}")
            print(f"   Papers: {result['papers_analyzed']}")
            print(f"   Fallback used: {result['metadata']['fallback_used']}")
            print(f"\n   Summary preview:")
            print(f"   {result['summary'][:200]}...\n")
            
        except Exception as e:
            print(f"\n❌ Test {i} FAILED: {e}")
            results.append({
                'scenario': scenario['name'],
                'success': False,
                'error': str(e)
            })
    
    # Print performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    orchestrator.get_performance_summary()
    
    # Print logger statistics
    logger = get_logger()
    stats = logger.get_statistics()
    
    print("\n" + "="*80)
    print("LOGGING STATISTICS")
    print("="*80)
    print(f"Total Events: {stats['total_events']}")
    print(f"Event Types: {stats['event_types']}")
    print(f"Total Errors: {stats['total_errors']}")
    if stats['total_errors'] > 0:
        print(f"Error Types: {stats['error_types']}")
    print(f"Session Duration: {stats['session_duration']:.1f} seconds")
    
    # Save logs
    log_path = orchestrator.save_logs()
    print(f"\n💾 Session logs saved to: {log_path}")
    
    # Test results summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} {'✗' if failed > 0 else ''}")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
    
    # Save test results
    os.makedirs('tests/results', exist_ok=True)
    with open('tests/results/production_test_results.json', 'w') as f:
        json.dump({
            'test_results': results,
            'logger_stats': stats,
            'performance_metrics': get_performance_tracker().get_metrics()
        }, f, indent=2)
    
    print(f"\n💾 Test results saved to: tests/results/production_test_results.json")
    
    return results

if __name__ == "__main__":
    results = test_production_system()
    
    print("\n" + "="*80)
    print("✅ PRODUCTION SYSTEM TEST COMPLETE")
    print("="*80)