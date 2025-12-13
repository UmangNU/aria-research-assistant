# experiments/train_rl.py
"""
Training Script for RL Agents
Trains DQN and Contextual Bandits on research queries
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

def generate_training_queries() -> list:
    """Generate diverse training queries - 100 unique queries"""
    queries = [
        # CS/ML queries (30 queries)
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
        "What are the applications of graph neural networks?",
        "Explain variational autoencoders",
        "What is few-shot learning?",
        "How does batch normalization work?",
        "What are capsule networks?",
        "Explain neural architecture search",
        "What is knowledge distillation?",
        "How do recurrent neural networks handle sequences?",
        "What are attention mechanisms in NLP?",
        "Explain diffusion models for image generation",
        "What is prompt engineering for large language models?",
        "How does BERT work?",
        "What are vision transformers?",
        "Explain contrastive language-image pretraining",
        "What is federated learning?",
        "How do neural ODEs work?",
        "What are normalizing flows?",
        "Explain adversarial training",
        "What is curriculum learning?",
        "How does gradient descent optimization work?",
        
        # Biology queries (20 queries)
        "What are recent advances in protein folding prediction?",
        "How does CRISPR gene editing work?",
        "What are the latest cancer immunotherapy developments?",
        "Explain epigenetics and gene expression",
        "What is single-cell sequencing?",
        "How does mRNA vaccine technology work?",
        "What are organoids in regenerative medicine?",
        "Explain synthetic biology approaches",
        "What is systems biology?",
        "How does the immune system recognize pathogens?",
        "What are induced pluripotent stem cells?",
        "Explain metabolomics in disease research",
        "What is the microbiome's role in health?",
        "How does DNA methylation work?",
        "What are CRISPR base editors?",
        "Explain protein-protein interactions",
        "What is structural biology?",
        "How do antibodies work?",
        "What are biosensors?",
        "Explain gene therapy approaches",
        
        # Physics queries (20 queries)
        "What are recent developments in quantum computing?",
        "Explain quantum entanglement",
        "What is topological quantum computing?",
        "How do superconductors work?",
        "What are quantum algorithms?",
        "Explain quantum error correction",
        "What is quantum supremacy?",
        "How do quantum gates work?",
        "What are photonic quantum computers?",
        "Explain ion trap quantum computing",
        "What is the quantum Zeno effect?",
        "How does quantum annealing work?",
        "What are topological insulators?",
        "Explain spintronics applications",
        "What is quantum teleportation?",
        "How do quantum sensors work?",
        "What are Majorana fermions?",
        "Explain quantum field theory basics",
        "What is the Higgs mechanism?",
        "How does dark matter detection work?",
        
        # Interdisciplinary queries (15 queries)
        "What is climate modeling?",
        "How do neural networks learn?",
        "What is machine learning?",
        "Explain artificial intelligence",
        "What are large language models?",
        "How does blockchain technology work?",
        "What is edge computing?",
        "Explain neuromorphic computing",
        "What is computational neuroscience?",
        "How does brain-computer interface work?",
        "What is digital twin technology?",
        "Explain augmented reality systems",
        "What is natural language processing?",
        "How does computer vision work?",
        "What is robotics and automation?",
        
        # Chemistry queries (15 queries)
        "What is computational chemistry?",
        "How does catalysis work?",
        "What are metal-organic frameworks?",
        "Explain organic synthesis methods",
        "What is materials science?",
        "How do batteries work?",
        "What is nanotechnology?",
        "Explain polymer chemistry",
        "What are quantum dots?",
        "How does drug discovery work?",
        "What is green chemistry?",
        "Explain electrochemistry applications",
        "What are coordination complexes?",
        "How does spectroscopy work?",
        "What is surface chemistry?",
    ]
    
    return queries
def train_rl_system(n_episodes: int = 50, save_freq: int = 10):
    """
    Train RL system on research queries
    
    Args:
        n_episodes: Number of training episodes
        save_freq: How often to save models
    """
    
    print("="*70)
    print("TRAINING RL SYSTEM")
    print("="*70)
    
    # Setup
    print("\n📚 Setting up system...")
    vs = VectorStore()
    
    # Load papers
    with open('data/papers/arxiv_papers.json', 'r') as f:
        papers = json.load(f)
    
    print(f"Using ALL {len(papers)} papers for comprehensive training")
    vs.add_papers(papers)  # Use ALL 1200 papers

    # Create RL orchestrator
    print("🤖 Initializing RL orchestrator...")
    rl_orch = RLOrchestrator(vs, use_rl=True)
    
    # Generate training queries
    training_queries = generate_training_queries()
    print(f"📝 Generated {len(training_queries)} training queries")
    
    # Training loop
    print(f"\n🎓 Starting training for {n_episodes} episodes...")
    print("="*70)
    
    results = []
    
    for episode in range(n_episodes):
        # Select query (cycle through)
        query = training_queries[episode % len(training_queries)]
        
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"Query: {query[:60]}...")
        print(f"{'='*70}")
        
        # Conduct research with training
        result = rl_orch.research(query, train=True)
        results.append(result)
        
        # Print progress
        stats = rl_orch.get_learning_stats()
        print(f"\n📊 Learning Progress:")
        print(f"   Avg Reward: {stats['avg_reward']:.3f}")
        print(f"   Avg Quality: {stats['avg_quality']:.3f}")
        print(f"   Recent Reward: {stats['recent_reward']:.3f}")
        print(f"   Recent Quality: {stats['recent_quality']:.3f}")
        print(f"   DQN Epsilon: {stats['dqn_epsilon']:.3f}")
        print(f"   Trend: {stats['quality_trend']}")
        
        # Save models periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs('models/checkpoints', exist_ok=True)
            rl_orch.save_models(
                f'models/checkpoints/dqn_episode_{episode+1}.pt',
                f'models/checkpoints/bandit_episode_{episode+1}.json'
            )
    
    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}\n")
    
    # Final statistics
    final_stats = rl_orch.get_learning_stats()
    print("📊 Final Statistics:")
    print(f"   Total Episodes: {final_stats['episodes']}")
    print(f"   Total Reward: {final_stats['total_reward']:.2f}")
    print(f"   Average Reward: {final_stats['avg_reward']:.3f}")
    print(f"   Average Quality: {final_stats['avg_quality']:.3f}")
    print(f"   Final Epsilon: {final_stats['dqn_epsilon']:.3f}")
    print(f"   Quality Trend: {final_stats['quality_trend']}")
    
    # Bandit statistics
    print(f"\n🎰 Bandit Arm Statistics:")
    for arm_stat in rl_orch.bandit.get_arm_stats():
        print(f"   {arm_stat['name']:20s}: "
              f"Pulls={arm_stat['total_pulls']:3d}, "
              f"Avg Reward={arm_stat['avg_reward']:.3f}, "
              f"Expected Value={arm_stat['expected_value']:.3f}")
    
    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    results_path = f'experiments/results/training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    # Convert results to JSON-serializable format
    results_json = []
    for r in results:
        results_json.append({
            'episode': r['episode'],
            'query': r['query'],
            'quality_score': float(r['quality_score']),
            'reward': float(r['reward']),
            'rl_action': int(r['rl_action']),
            'bandit_strategy': r['bandit_strategy']
        })
    
    with open(results_path, 'w') as f:
        json.dump({
            'training_results': results_json,
            'final_stats': final_stats,
            'n_episodes': n_episodes
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Plot learning curves
    plot_learning_curves(results)
    
    # Save final models
    rl_orch.save_models(
        'models/dqn_final.pt',
        'models/bandit_final.json'
    )
    
    return results, final_stats

def plot_learning_curves(results: list):
    """Plot learning curves"""
    
    episodes = [r['episode'] for r in results]
    rewards = [r['reward'] for r in results]
    qualities = [r['quality_score'] for r in results]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot rewards
    ax1.plot(episodes, rewards, alpha=0.3, label='Episode Reward')
    
    # Moving average
    window = 5
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('RL Training: Reward over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot quality scores
    ax2.plot(episodes, qualities, alpha=0.3, label='Quality Score')
    
    if len(qualities) >= window:
        moving_avg = np.convolve(qualities, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(qualities)), moving_avg, 'g-', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Quality Score')
    ax2.set_title('RL Training: Quality Score over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('experiments/results', exist_ok=True)
    plot_path = f'experiments/results/learning_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Learning curves saved to: {plot_path}")
    
    plt.close()

if __name__ == "__main__":
    # Run training
    results, stats = train_rl_system(n_episodes=1000, save_freq=100)
    
    print("\n" + "="*70)
    print("🎉 Training session complete!")
    print("="*70)