# experiments/baseline_comparison.py
"""
Baseline Comparison: Random vs RL
Proves that RL improves over random agent selection
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.vector_store import VectorStore
from src.rl.rl_orchestrator import RLOrchestrator
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_baseline_comparison():
    """Compare random baseline vs trained RL agent"""
    
    print("="*80)
    print("BASELINE COMPARISON: Random vs RL")
    print("="*80)
    
    # Load data
    print("\n📚 Loading papers...")
    vs = VectorStore()
    with open('data/papers/arxiv_papers.json', 'r') as f:
        papers = json.load(f)
    vs.add_papers(papers)
    
    # Test queries
    test_queries = [
        "What are recent advances in deep reinforcement learning?",
        "How do transformer models work?",
        "What is protein folding prediction?",
        "Explain quantum computing",
        "What is climate modeling?",
        "How does CRISPR work?",
        "What are neural networks?",
        "Explain machine learning",
        "What is computer vision?",
        "How does NLP work?",
    ] * 10  # 100 test queries
    
    print(f"Running {len(test_queries)} test episodes for each condition...\n")
    
    # 1. Random baseline (no RL)
    print("="*80)
    print("CONDITION 1: Random Baseline (No Learning)")
    print("="*80)
    baseline_orch = RLOrchestrator(vs, use_rl=False)
    baseline_results = []
    
    for i, query in enumerate(test_queries):
        if (i + 1) % 20 == 0:
            print(f"  Baseline: {i+1}/{len(test_queries)} complete...")
        result = baseline_orch.research(query, train=False)
        baseline_results.append(result['quality_score'])
    
    baseline_avg = np.mean(baseline_results)
    baseline_std = np.std(baseline_results)
    
    # 2. Trained RL agent
    print("\n" + "="*80)
    print("CONDITION 2: Trained RL Agent")
    print("="*80)
    rl_orch = RLOrchestrator(vs, use_rl=True)
    
    # Load trained models
    print("Loading trained RL models...")
    rl_orch.load_models('models/dqn_final_1000ep.pt', 'models/bandit_final_1000ep.json')
    
    rl_results = []
    for i, query in enumerate(test_queries):
        if (i + 1) % 20 == 0:
            print(f"  RL Agent: {i+1}/{len(test_queries)} complete...")
        result = rl_orch.research(query, train=False)
        rl_results.append(result['quality_score'])
    
    rl_avg = np.mean(rl_results)
    rl_std = np.std(rl_results)
    
    # Statistical analysis
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nBaseline (Random):")
    print(f"  Mean Quality: {baseline_avg:.4f}")
    print(f"  Std Dev: {baseline_std:.4f}")
    
    print(f"\nRL Agent (Trained):")
    print(f"  Mean Quality: {rl_avg:.4f}")
    print(f"  Std Dev: {rl_std:.4f}")
    
    improvement = ((rl_avg - baseline_avg) / baseline_avg) * 100
    print(f"\nImprovement: {improvement:+.2f}%")
    
    # T-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(rl_results, baseline_results)
    print(f"\nStatistical Significance:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  Result: ✓ Statistically significant (p < 0.05)")
    else:
        print(f"  Result: Not statistically significant")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((baseline_std**2 + rl_std**2) / 2)
    cohens_d = (rl_avg - baseline_avg) / pooled_std
    print(f"  Cohen's d: {cohens_d:.4f}")
    
    # Visualization
    plot_comparison(baseline_results, rl_results, baseline_avg, rl_avg)
    
    # Save results
    results = {
        'baseline': {
            'mean': float(baseline_avg),
            'std': float(baseline_std),
            'results': [float(x) for x in baseline_results]
        },
        'rl_agent': {
            'mean': float(rl_avg),
            'std': float(rl_std),
            'results': [float(x) for x in rl_results]
        },
        'improvement_percent': float(improvement),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d)
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'experiments/results/baseline_comparison_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: experiments/results/baseline_comparison_{timestamp}.json")
    
    return results

def plot_comparison(baseline, rl, baseline_avg, rl_avg):
    """Create comparison visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Box plot comparison
    ax1 = axes[0, 0]
    ax1.boxplot([baseline, rl], labels=['Random Baseline', 'RL Agent'])
    ax1.set_ylabel('Quality Score')
    ax1.set_title('Quality Score Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram comparison
    ax2 = axes[0, 1]
    ax2.hist(baseline, bins=30, alpha=0.5, label='Random Baseline', color='red')
    ax2.hist(rl, bins=30, alpha=0.5, label='RL Agent', color='green')
    ax2.axvline(baseline_avg, color='red', linestyle='--', linewidth=2)
    ax2.axvline(rl_avg, color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Quality Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Quality Score Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bar chart with error bars
    ax3 = axes[1, 0]
    means = [baseline_avg, rl_avg]
    stds = [np.std(baseline), np.std(rl)]
    x_pos = [0, 1]
    ax3.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
            color=['red', 'green'], edgecolor='black', linewidth=2)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Random Baseline', 'RL Agent'])
    ax3.set_ylabel('Mean Quality Score')
    ax3.set_title('Mean Performance Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Sequential performance
    ax4 = axes[1, 1]
    ax4.plot(baseline, alpha=0.5, color='red', label='Random Baseline')
    ax4.plot(rl, alpha=0.5, color='green', label='RL Agent')
    # Moving averages
    window = 10
    baseline_ma = np.convolve(baseline, np.ones(window)/window, mode='valid')
    rl_ma = np.convolve(rl, np.ones(window)/window, mode='valid')
    ax4.plot(range(window-1, len(baseline)), baseline_ma, color='darkred', linewidth=2)
    ax4.plot(range(window-1, len(rl)), rl_ma, color='darkgreen', linewidth=2)
    ax4.set_xlabel('Test Episode')
    ax4.set_ylabel('Quality Score')
    ax4.set_title('Sequential Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'experiments/results/baseline_comparison_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plots saved")
    
    plt.close()

if __name__ == "__main__":
    results = run_baseline_comparison()
    print("\n" + "="*80)
    print("✅ Baseline comparison complete!")
    print("="*80)