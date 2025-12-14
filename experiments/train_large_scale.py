# experiments/train_large_scale.py
"""
Large-Scale RL Training - 1000 Episodes
For comprehensive RL demonstration
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
import time

def generate_training_queries() -> list:
    """Generate 100 diverse training queries"""
    queries = [
        # CS/ML (30)
        "What are recent advances in deep reinforcement learning?",
        "How do transformer models work?",
        "What are the latest developments in computer vision?",
        "Explain contrastive learning methods",
        "What is self-supervised learning?",
        "How does attention mechanism work in neural networks?",
        "What are generative adversarial networks?",
        "Explain meta-learning approaches",
        "What is transfer learning in deep learning?",
        "How do convolutional neural networks process images?",
        "What are graph neural networks?",
        "Explain variational autoencoders",
        "What is few-shot learning?",
        "How does batch normalization work?",
        "What are capsule networks?",
        "Explain neural architecture search",
        "What is knowledge distillation?",
        "How do RNNs handle sequences?",
        "What are NLP attention mechanisms?",
        "Explain diffusion models",
        "What is prompt engineering?",
        "How does BERT work?",
        "What are vision transformers?",
        "Explain CLIP models",
        "What is federated learning?",
        "How do neural ODEs work?",
        "What are normalizing flows?",
        "Explain adversarial training",
        "What is curriculum learning?",
        "How does gradient descent work?",
        
        # Biology (20)
        "What are recent advances in protein folding?",
        "How does CRISPR work?",
        "What are cancer immunotherapy developments?",
        "Explain epigenetics",
        "What is single-cell sequencing?",
        "How does mRNA vaccine work?",
        "What are organoids?",
        "Explain synthetic biology",
        "What is systems biology?",
        "How does immune system work?",
        "What are stem cells?",
        "Explain metabolomics",
        "What is microbiome role?",
        "How does DNA methylation work?",
        "What are base editors?",
        "Explain protein interactions",
        "What is structural biology?",
        "How do antibodies work?",
        "What are biosensors?",
        "Explain gene therapy",
        
        # Physics (20)
        "What is quantum computing?",
        "Explain quantum entanglement",
        "What is topological computing?",
        "How do superconductors work?",
        "What are quantum algorithms?",
        "Explain error correction",
        "What is quantum supremacy?",
        "How do quantum gates work?",
        "What are photonic computers?",
        "Explain ion traps",
        "What is quantum Zeno?",
        "How does annealing work?",
        "What are topological insulators?",
        "Explain spintronics",
        "What is teleportation?",
        "How do quantum sensors work?",
        "What are Majorana fermions?",
        "Explain field theory",
        "What is Higgs mechanism?",
        "How is dark matter detected?",
        
        # Interdisciplinary (15)
        "What is climate modeling?",
        "How do neural networks learn?",
        "What is machine learning?",
        "Explain AI",
        "What are LLMs?",
        "How does blockchain work?",
        "What is edge computing?",
        "Explain neuromorphic computing?",
        "What is computational neuroscience?",
        "How does BCI work?",
        "What is digital twin?",
        "Explain AR systems",
        "What is NLP?",
        "How does CV work?",
        "What is robotics?",
        
        # Chemistry (15)
        "What is computational chemistry?",
        "How does catalysis work?",
        "What are MOFs?",
        "Explain synthesis methods",
        "What is materials science?",
        "How do batteries work?",
        "What is nanotechnology?",
        "Explain polymers",
        "What are quantum dots?",
        "How does drug discovery work?",
        "What is green chemistry?",
        "Explain electrochemistry",
        "What are complexes?",
        "How does spectroscopy work?",
        "What is surface chemistry?",
    ]
    return queries

def train_large_scale(n_episodes: int = 1000):
    """
    Train RL system at scale
    
    Args:
        n_episodes: Number of training episodes (default: 1000)
    """
    
    print("="*80)
    print(f"LARGE-SCALE RL TRAINING - {n_episodes} EPISODES")
    print("="*80)
    
    start_time = time.time()
    
    # Setup
    print("\n📚 Loading ALL papers...")
    vs = VectorStore()
    
    with open('data/papers/arxiv_papers.json', 'r') as f:
        papers = json.load(f)
    
    print(f"Found {len(papers)} papers")
    print("Indexing all papers (this takes a moment)...")
    vs.add_papers(papers)  # ALL papers
    
    print(f"\n🤖 Initializing RL orchestrator...")
    rl_orch = RLOrchestrator(vs, use_rl=True)
    
    # Generate queries
    training_queries = generate_training_queries()
    print(f"📝 Using {len(training_queries)} unique training queries")
    print(f"Will cycle through them for {n_episodes} episodes\n")
    
    # Training loop
    print(f"🎓 Starting training for {n_episodes} episodes...")
    print("="*80)
    
    results = []
    checkpoint_freq = 100
    
    for episode in range(n_episodes):
        # Select query (cycle through)
        query = training_queries[episode % len(training_queries)]
        
        # Progress indicator every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"\n{'='*80}")
            print(f"Episode {episode + 1}/{n_episodes}")
            elapsed = time.time() - start_time
            avg_time = elapsed / (episode + 1)
            eta = avg_time * (n_episodes - episode - 1)
            print(f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
            stats = rl_orch.get_learning_stats()
            print(f"Avg Quality: {stats['avg_quality']:.3f} | Recent: {stats['recent_quality']:.3f}")
            print(f"Trend: {stats['quality_trend']} | Epsilon: {stats['dqn_epsilon']:.3f}")
            print(f"{'='*80}")
        
        # Conduct research
        result = rl_orch.research(query, train=True)
        results.append({
            'episode': result['episode'],
            'query': result['query'],
            'quality_score': float(result['quality_score']),
            'reward': float(result['reward']),
            'rl_action': int(result['rl_action']),
            'bandit_strategy': result['bandit_strategy']
        })
        
        # Save checkpoints
        if (episode + 1) % checkpoint_freq == 0:
            os.makedirs('models/checkpoints', exist_ok=True)
            rl_orch.save_models(
                f'models/checkpoints/dqn_episode_{episode+1}.pt',
                f'models/checkpoints/bandit_episode_{episode+1}.json'
            )
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("✅ Training Complete!")
    print(f"{'='*80}\n")
    
    # Final statistics
    final_stats = rl_orch.get_learning_stats()
    print("📊 Final Statistics:")
    print(f"   Total Episodes: {final_stats['episodes']}")
    print(f"   Total Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Avg Time per Episode: {total_time/n_episodes:.2f} seconds")
    print(f"   Total Reward: {final_stats['total_reward']:.2f}")
    print(f"   Average Reward: {final_stats['avg_reward']:.3f}")
    print(f"   Average Quality: {final_stats['avg_quality']:.3f}")
    print(f"   Recent Quality (last 20): {final_stats['recent_quality']:.3f}")
    print(f"   Quality Trend: {final_stats['quality_trend']}")
    print(f"   Final Epsilon: {final_stats['dqn_epsilon']:.3f}")
    
    # Bandit statistics
    print(f"\n🎰 Bandit Arm Statistics:")
    for arm_stat in rl_orch.bandit.get_arm_stats():
        print(f"   {arm_stat['name']:20s}: "
              f"Pulls={arm_stat['total_pulls']:4d}, "
              f"Avg Reward={arm_stat['avg_reward']:.3f}, "
              f"Expected Value={arm_stat['expected_value']:.3f}")
    
    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'experiments/results/large_scale_training_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump({
            'training_results': results,
            'final_stats': final_stats,
            'n_episodes': n_episodes,
            'total_time_seconds': total_time,
            'papers_used': len(papers)
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Plot learning curves
    plot_large_scale_curves(results, n_episodes)
    
    # Save final models
    rl_orch.save_models(
        'models/dqn_final_1000ep.pt',
        'models/bandit_final_1000ep.json'
    )
    
    return results, final_stats

def plot_large_scale_curves(results: list, n_episodes: int):
    """Plot comprehensive learning curves"""
    
    episodes = [r['episode'] for r in results]
    rewards = [r['reward'] for r in results]
    qualities = [r['quality_score'] for r in results]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Reward over time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(episodes, rewards, alpha=0.2, color='blue', linewidth=0.5)
    
    # Moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode MA')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'RL Training: Reward over {n_episodes} Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Quality over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(episodes, qualities, alpha=0.2, color='green', linewidth=0.5)
    
    if len(qualities) >= window:
        moving_avg = np.convolve(qualities, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(qualities)), moving_avg, 'orange', linewidth=2, label=f'{window}-Episode MA')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Quality Score')
    ax2.set_title(f'Quality Score over {n_episodes} Episodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reward Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Quality distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(qualities, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(np.mean(qualities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(qualities):.3f}')
    ax4.set_xlabel('Quality Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Quality Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Learning progress (binned)
    ax5 = plt.subplot(2, 3, 5)
    bin_size = n_episodes // 10
    binned_rewards = [np.mean(rewards[i:i+bin_size]) for i in range(0, len(rewards), bin_size)]
    ax5.plot(range(len(binned_rewards)), binned_rewards, 'o-', linewidth=2, markersize=8)
    ax5.set_xlabel('Training Phase (10 bins)')
    ax5.set_ylabel('Average Reward')
    ax5.set_title('Learning Progress Across Training')
    ax5.grid(True, alpha=0.3)
    
    # 6. Action distribution
    ax6 = plt.subplot(2, 3, 6)
    actions = [r['rl_action'] for r in results]
    action_counts = [actions.count(i) for i in range(6)]
    ax6.bar(range(6), action_counts, alpha=0.7, color='purple', edgecolor='black')
    ax6.set_xlabel('Action ID')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Action Selection Distribution')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'experiments/results/large_scale_curves_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Learning curves saved to: {plot_path}")
    
    plt.close()

if __name__ == "__main__":
    print("\n🚀 Starting large-scale RL training...")
    print("This will take approximately 10-15 minutes\n")
    
    results, stats = train_large_scale(n_episodes=1000)
    
    print("\n" + "="*80)
    print("🎉 Large-scale training complete!")
    print("="*80)