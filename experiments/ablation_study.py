# experiments/ablation_study.py
"""
Complete Ablation Study: Test impact of each RL component
Tests all combinations to prove each component contributes
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.vector_store import VectorStore
from src.rl.rl_orchestrator import RLOrchestrator
from src.rl.dqn import DQNAgent
from src.rl.bandit import SourceSelectionBandit
from src.rl.state_builder import StateBuilder
from src.rl.replay_buffer import ReplayBuffer
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

class AblationOrchestrator:
    """Custom orchestrator for ablation testing"""
    
    def __init__(self, vector_store, config_name: str):
        """
        Initialize with specific ablation configuration
        
        Args:
            vector_store: Vector store
            config_name: 'random', 'dqn_only', 'bandit_only', or 'full_rl'
        """
        self.base_orch = RLOrchestrator(vector_store, use_rl=True)
        self.config_name = config_name
        
        # Store original methods
        self.original_dqn_select = self.base_orch.dqn_agent.select_action
        self.original_bandit_select = self.base_orch.bandit.select_arm
        
    def research(self, query: str, train: bool = True):
        """Research with specific ablation config"""
        
        if self.config_name == 'random':
            # Random actions
            def random_dqn(state, explore=True):
                return np.random.randint(0, 6)
            
            def random_bandit():
                return np.random.randint(0, 5)
            
            self.base_orch.dqn_agent.select_action = random_dqn
            self.base_orch.bandit.select_arm = random_bandit
            
            result = self.base_orch.research(query, train=False)
            
            # Restore
            self.base_orch.dqn_agent.select_action = self.original_dqn_select
            self.base_orch.bandit.select_arm = self.original_bandit_select
            
        elif self.config_name == 'dqn_only':
            # DQN learns, Bandit fixed to arm 0
            def fixed_bandit():
                return 0
            
            self.base_orch.bandit.select_arm = fixed_bandit
            
            result = self.base_orch.research(query, train=train)
            
            # Restore
            self.base_orch.bandit.select_arm = self.original_bandit_select
            
        elif self.config_name == 'bandit_only':
            # Bandit learns, DQN fixed to action 3
            def fixed_dqn(state, explore=True):
                return 3
            
            self.base_orch.dqn_agent.select_action = fixed_dqn
            
            result = self.base_orch.research(query, train=train)
            
            # Restore
            self.base_orch.dqn_agent.select_action = self.original_dqn_select
            
        else:  # full_rl
            result = self.base_orch.research(query, train=train)
        
        return result

def run_ablation_study(n_episodes_per_config: int = 200):
    """Run complete ablation study"""
    
    print("="*80)
    print(f"COMPLETE ABLATION STUDY - {n_episodes_per_config} episodes per config")
    print("="*80)
    
    # Load data once
    print("\n📚 Loading papers...")
    vs = VectorStore()
    with open('data/papers/arxiv_papers.json', 'r') as f:
        papers = json.load(f)
    vs.add_papers(papers)
    
    queries = [
        "What are recent advances in deep learning?",
        "How do transformers work?",
        "What is reinforcement learning?",
        "Explain computer vision",
        "What is NLP?",
        "How does protein folding work?",
        "What is quantum computing?",
        "Explain neural networks",
        "What is machine learning?",
        "How does AI work?",
    ]
    
    configs = ['random', 'dqn_only', 'bandit_only', 'full_rl']
    config_names = {
        'random': 'Random Baseline',
        'dqn_only': 'DQN Only',
        'bandit_only': 'Bandit Only',
        'full_rl': 'Full RL (DQN + Bandit)'
    }
    
    all_results = {}
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"CONFIG {i}/{len(configs)}: {config_names[config]}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        orch = AblationOrchestrator(vs, config)
        
        episode_qualities = []
        episode_rewards = []
        
        for episode in range(n_episodes_per_config):
            query = queries[episode % len(queries)]
            
            train_flag = (config != 'random')  # Don't train random
            result = orch.research(query, train=train_flag)
            
            episode_qualities.append(result['quality_score'])
            episode_rewards.append(result['reward'])
            
            if (episode + 1) % 50 == 0:
                recent_q = np.mean(episode_qualities[-20:]) if len(episode_qualities) >= 20 else np.mean(episode_qualities)
                print(f"  Episode {episode+1}/{n_episodes_per_config} | "
                      f"Avg: {np.mean(episode_qualities):.3f} | "
                      f"Recent: {recent_q:.3f}")
        
        elapsed = time.time() - start_time
        
        all_results[config] = {
            'name': config_names[config],
            'qualities': episode_qualities,
            'rewards': episode_rewards,
            'mean_quality': np.mean(episode_qualities),
            'std_quality': np.std(episode_qualities),
            'final_quality': np.mean(episode_qualities[-20:]),
            'initial_quality': np.mean(episode_qualities[:20]),
            'time_seconds': elapsed
        }
        
        print(f"\n✓ {config_names[config]} complete:")
        print(f"   Mean Quality: {all_results[config]['mean_quality']:.4f}")
        print(f"   Final Quality: {all_results[config]['final_quality']:.4f}")
        print(f"   Time: {elapsed/60:.1f} minutes")
    
    # Comprehensive analysis
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    print("\n" + f"{'Configuration':<30} {'Mean':>10} {'Final':>10} {'Improvement':>12}")
    print("-"*80)
    
    baseline_mean = all_results['random']['mean_quality']
    
    for config in configs:
        data = all_results[config]
        improvement = ((data['mean_quality'] - baseline_mean) / baseline_mean) * 100
        print(f"{data['name']:<30} {data['mean_quality']:>10.4f} {data['final_quality']:>10.4f} {improvement:>11.2f}%")
    
    # Statistical tests
    print("\n" + "="*80)
    print("PAIRWISE STATISTICAL TESTS")
    print("="*80)
    
    from scipy import stats
    
    comparisons = [
        ('random', 'dqn_only'),
        ('random', 'bandit_only'),
        ('random', 'full_rl'),
        ('dqn_only', 'full_rl'),
        ('bandit_only', 'full_rl'),
    ]
    
    for config1, config2 in comparisons:
        t_stat, p_value = stats.ttest_ind(
            all_results[config1]['qualities'],
            all_results[config2]['qualities']
        )
        
        name1 = all_results[config1]['name']
        name2 = all_results[config2]['name']
        
        print(f"\n{name1} vs {name2}:")
        print(f"  t-stat: {t_stat:>8.4f} | p-value: {p_value:.6f} | {'✓ Sig' if p_value < 0.05 else '✗ NS'}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'experiments/results/ablation_study_{timestamp}.json'
    
    json_results = {
        config: {
            'name': data['name'],
            'mean_quality': float(data['mean_quality']),
            'std_quality': float(data['std_quality']),
            'final_quality': float(data['final_quality']),
            'initial_quality': float(data['initial_quality']),
            'qualities': [float(x) for x in data['qualities']],
            'rewards': [float(x) for x in data['rewards']]
        }
        for config, data in all_results.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Plot
    plot_comprehensive_ablation(all_results, timestamp)
    
    return all_results

def plot_comprehensive_ablation(all_results, timestamp):
    """Create comprehensive ablation plots"""
    
    fig = plt.figure(figsize=(16, 12))
    
    configs = ['random', 'dqn_only', 'bandit_only', 'full_rl']
    colors = {'random': 'red', 'dqn_only': 'blue', 'bandit_only': 'orange', 'full_rl': 'green'}
    
    # 1. Learning curves (all configs)
    ax1 = plt.subplot(3, 3, 1)
    for config in configs:
        qualities = all_results[config]['qualities']
        window = 20
        if len(qualities) >= window:
            ma = np.convolve(qualities, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(qualities)), ma,
                    color=colors[config], linewidth=2, label=all_results[config]['name'])
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Quality (20-ep MA)')
    ax1.set_title('Learning Curves')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plots
    ax2 = plt.subplot(3, 3, 2)
    data_for_box = [all_results[k]['qualities'] for k in configs]
    labels_short = ['Random', 'DQN', 'Bandit', 'Full RL']
    bp = ax2.boxplot(data_for_box, labels=labels_short, patch_artist=True)
    for patch, config in zip(bp['boxes'], configs):
        patch.set_facecolor(colors[config])
        patch.set_alpha(0.6)
    ax2.set_ylabel('Quality Score')
    ax2.set_title('Quality Distributions')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Mean comparison
    ax3 = plt.subplot(3, 3, 3)
    means = [all_results[k]['mean_quality'] for k in configs]
    stds = [all_results[k]['std_quality'] for k in configs]
    bars = ax3.bar(range(4), means, yerr=stds, capsize=8, alpha=0.7,
                   color=[colors[k] for k in configs], edgecolor='black', linewidth=2)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(labels_short)
    ax3.set_ylabel('Mean Quality')
    ax3.set_title('Mean Performance')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, m in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2., m, f'{m:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Improvements
    ax4 = plt.subplot(3, 3, 4)
    baseline = all_results['random']['mean_quality']
    improvements = [((all_results[k]['mean_quality'] - baseline) / baseline) * 100 for k in configs]
    bars = ax4.bar(range(4), improvements, alpha=0.7,
                   color=[colors[k] for k in configs], edgecolor='black', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(labels_short)
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Relative Performance')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=9)
    
    # 5. Initial vs Final (learning progress)
    ax5 = plt.subplot(3, 3, 5)
    initial = [all_results[k]['initial_quality'] for k in configs]
    final = [all_results[k]['final_quality'] for k in configs]
    x = np.arange(4)
    width = 0.35
    ax5.bar(x - width/2, initial, width, label='Initial (first 20)', alpha=0.7, color='lightgray')
    ax5.bar(x + width/2, final, width, label='Final (last 20)', alpha=0.7,
            color=[colors[k] for k in configs])
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels_short)
    ax5.set_ylabel('Quality Score')
    ax5.set_title('Learning: Initial vs Final')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Histograms overlaid
    ax6 = plt.subplot(3, 3, 6)
    for config in configs:
        ax6.hist(all_results[config]['qualities'], bins=30, alpha=0.4,
                label=all_results[config]['name'], color=colors[config])
    ax6.set_xlabel('Quality Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Quality Distributions Overlaid')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Reward curves
    ax7 = plt.subplot(3, 3, 7)
    for config in configs:
        rewards = all_results[config]['rewards']
        window = 20
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax7.plot(range(window-1, len(rewards)), ma,
                    color=colors[config], linewidth=2, label=all_results[config]['name'])
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Reward (20-ep MA)')
    ax7.set_title('Reward Curves')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Variance comparison
    ax8 = plt.subplot(3, 3, 8)
    variances = [all_results[k]['std_quality']**2 for k in configs]
    bars = ax8.bar(range(4), variances, alpha=0.7,
                   color=[colors[k] for k in configs], edgecolor='black', linewidth=2)
    ax8.set_xticks(range(4))
    ax8.set_xticklabels(labels_short)
    ax8.set_ylabel('Variance')
    ax8.set_title('Performance Stability (Lower = More Stable)')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Component contribution
    ax9 = plt.subplot(3, 3, 9)
    baseline = all_results['random']['mean_quality']
    dqn_contribution = all_results['dqn_only']['mean_quality'] - baseline
    bandit_contribution = all_results['bandit_only']['mean_quality'] - baseline
    combined = all_results['full_rl']['mean_quality'] - baseline
    
    contributions = [0, dqn_contribution, bandit_contribution, combined]
    labels_contrib = ['Baseline', 'DQN\nContribution', 'Bandit\nContribution', 'Combined\nEffect']
    bars = ax9.bar(range(4), contributions, alpha=0.7,
                   color=['gray', 'blue', 'orange', 'green'], edgecolor='black', linewidth=2)
    ax9.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax9.set_xticks(range(4))
    ax9.set_xticklabels(labels_contrib, fontsize=9)
    ax9.set_ylabel('Quality Improvement over Baseline')
    ax9.set_title('Component Contributions')
    ax9.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, contributions):
        if val != 0:
            ax9.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:+.3f}', ha='center',
                    va='bottom' if val > 0 else 'top',
                    fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'experiments/results/ablation_comprehensive_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive ablation plots saved")
    plt.close()

def train_config(vs, config_name: str, n_episodes: int, queries: list):
    """Train a specific configuration"""
    
    orch = AblationOrchestrator(vs, config_name)
    
    qualities = []
    rewards = []
    
    for episode in range(n_episodes):
        query = queries[episode % len(queries)]
        train_flag = (config_name != 'random')
        
        result = orch.research(query, train=train_flag)
        qualities.append(result['quality_score'])
        rewards.append(result['reward'])
        
        if (episode + 1) % 50 == 0:
            recent = np.mean(qualities[-20:]) if len(qualities) >= 20 else np.mean(qualities)
            print(f"  Episode {episode+1}/{n_episodes} | Avg: {np.mean(qualities):.3f} | Recent: {recent:.3f}")
    
    return {
        'qualities': qualities,
        'rewards': rewards,
        'mean_quality': np.mean(qualities),
        'std_quality': np.std(qualities),
        'final_quality': np.mean(qualities[-20:]),
        'initial_quality': np.mean(qualities[:20])
    }

def main():
    """Main execution"""
    
    start_time = time.time()
    
    results = run_ablation_study(n_episodes_per_config=200)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    from scipy import stats
    
    baseline = results['random']['mean_quality']
    
    print(f"\n{'Config':<20} {'Mean':>10} {'Stdev':>10} {'Improve':>10} {'p-value':>12}")
    print("-"*80)
    
    for config in ['random', 'dqn_only', 'bandit_only', 'full_rl']:
        data = results[config]
        improvement = ((data['mean_quality'] - baseline) / baseline) * 100
        
        # T-test vs baseline
        if config != 'random':
            t, p = stats.ttest_ind(data['qualities'], results['random']['qualities'])
            p_str = f"{p:.6f}"
        else:
            p_str = "---"
        
        print(f"{data['name']:<20} {data['mean_quality']:>10.4f} {data['std_quality']:>10.4f} "
              f"{improvement:>9.2f}% {p_str:>12}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ Complete ablation study finished in {elapsed/60:.1f} minutes")
    print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    results = main()