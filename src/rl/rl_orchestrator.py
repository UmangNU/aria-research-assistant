# src/rl/rl_orchestrator.py
"""
RL-Enhanced Orchestrator
Integrates reinforcement learning with the agent system
"""

from typing import Dict, Any, List
import numpy as np
from src.agents.orchestrator import AgentOrchestrator
from src.rl.dqn import DQNAgent
from src.rl.bandit import SourceSelectionBandit
from src.rl.state_builder import StateBuilder
from src.rl.replay_buffer import ReplayBuffer
from src.rag.vector_store import VectorStore

class RLOrchestrator:
    """
    Orchestrator with Reinforcement Learning
    Learns optimal research strategies through experience
    """
    
    def __init__(self, vector_store: VectorStore, use_rl: bool = True):
        """
        Initialize RL Orchestrator
        
        Args:
            vector_store: Vector store for paper retrieval
            use_rl: Whether to use RL (True) or random baseline (False)
        """
        # Base orchestrator
        self.orchestrator = AgentOrchestrator(vector_store)
        self.use_rl = use_rl
        
        # RL Components
        self.state_builder = StateBuilder()
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Action space: 
        # 0: top_k=3, depth=shallow
        # 1: top_k=5, depth=shallow
        # 2: top_k=5, depth=moderate
        # 3: top_k=10, depth=moderate
        # 4: top_k=10, depth=deep
        # 5: top_k=15, depth=deep
        self.action_space = [
            {'top_k': 3, 'depth': 'shallow', 'style': 'concise'},
            {'top_k': 5, 'depth': 'shallow', 'style': 'concise'},
            {'top_k': 5, 'depth': 'moderate', 'style': 'detailed'},
            {'top_k': 10, 'depth': 'moderate', 'style': 'detailed'},
            {'top_k': 10, 'depth': 'deep', 'style': 'detailed'},
            {'top_k': 15, 'depth': 'deep', 'style': 'technical'}
        ]
        
        # DQN Agent
        state_dim = self.state_builder.get_state_dim()
        action_dim = len(self.action_space)
        self.dqn_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
        
        # Contextual Bandit for source selection
        self.bandit = SourceSelectionBandit(n_arms=5)
        
        # Training statistics
        self.episode = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.episode_qualities = []
        
    def research(self, query: str, train: bool = True) -> Dict[str, Any]:
        """
        Conduct research with RL
        
        Args:
            query: Research question
            train: Whether to train RL agents
            
        Returns:
            Research results with RL metrics
        """
        
        # Step 1: Analyze query
        query_analysis = self.orchestrator.query_analyzer.execute({'query': query})
        
        # Step 2: Initial source discovery (get more papers)
        initial_sources = self.orchestrator.source_discovery.execute({
            'query': query,
            'keywords': query_analysis['keywords'],
            'domain': query_analysis['domain'],
            'top_k': 20
        })
        
        # Step 3: Build state for RL
        state = self.state_builder.build_state(
            query_analysis=query_analysis,
            sources=initial_sources,
            papers_analyzed=0,
            current_quality=0.0
        )
        
        # Step 4: DQN selects configuration
        if self.use_rl:
            action = self.dqn_agent.select_action(state, explore=train)
        else:
            action = np.random.randint(0, len(self.action_space))
        
        config = self.action_space[action]
        
        print(f"\n🧠 DQN Selected Action {action}: {config}")
        
        # Step 5: Bandit selects strategy
        if self.use_rl:
            bandit_arm = self.bandit.select_arm()
        else:
            bandit_arm = 0
        
        print(f"🎰 Bandit Selected Strategy: {self.bandit.arm_names[bandit_arm]}")
        
        # Apply bandit strategy
        papers = self.bandit.apply_strategy(bandit_arm, initial_sources['papers'])
        
        # Create config with RL-selected parameters
        rl_config = {
            'top_k': config['top_k'],
            'depth': config['depth'],
            'style': config['style']
        }
        
        # Step 6: Run FULL orchestrator pipeline
        print(f"\n🔬 Starting full research pipeline...")
        result = self.orchestrator.research(query, config=rl_config)
        
        quality_score = result['quality_score']
        reward = result['reward']
        
        print(f"\n📊 RL Results:")
        print(f"   Quality Score: {quality_score:.3f}")
        print(f"   RL Reward: {reward:.3f}")
        
        # Step 7: RL Training
        if train and self.use_rl:
            next_state = self.state_builder.build_state(
                query_analysis=query_analysis,
                sources=initial_sources,
                papers_analyzed=result['papers_analyzed'],
                current_quality=quality_score
            )
            
            self.replay_buffer.add(state, action, reward, next_state, True)
            
            if self.replay_buffer.is_ready(self.dqn_agent.batch_size):
                loss = self.dqn_agent.train_step(self.replay_buffer)
                if self.episode % 10 == 0 and self.episode > 0:
                    print(f"🎓 DQN Training - Loss: {loss:.4f}")
            
            self.bandit.update(bandit_arm, quality_score)
            self.state_builder.update_history(quality_score, reward)
            
            self.episode += 1
            self.total_reward += reward
            self.episode_rewards.append(reward)
            self.episode_qualities.append(quality_score)
        
        # Add RL info to results
        result['rl_action'] = action
        result['rl_config'] = config
        result['bandit_strategy'] = self.bandit.arm_names[bandit_arm]
        result['episode'] = self.episode
        result['dqn_stats'] = self.dqn_agent.get_stats() if self.use_rl else {}
        result['bandit_stats'] = self.bandit.get_arm_stats() if self.use_rl else []
        
        return result
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = {
            'episodes': self.episode,
            'total_reward': self.total_reward,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_quality': np.mean(self.episode_qualities) if self.episode_qualities else 0.0,
            'recent_reward': np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) >= 20 else 0.0,
            'recent_quality': np.mean(self.episode_qualities[-20:]) if len(self.episode_qualities) >= 20 else 0.0,
            'reward_trend': self._calculate_trend(self.episode_rewards),
            'quality_trend': self._calculate_trend(self.episode_qualities),
            'dqn_epsilon': self.dqn_agent.epsilon,
            'bandit_exploration': self.bandit.get_exploration_rate(),
            'best_bandit_arm': self.bandit.get_best_arm()
        }
        return stats
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if trend is improving, declining, or stable"""
        if len(values) < 10:
            return "insufficient_data"
        
        recent = values[-10:]
        earlier = values[-20:-10] if len(values) >= 20 else values[:-10]
        
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        diff = recent_avg - earlier_avg
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    def save_models(self, dqn_path: str, bandit_path: str):
        """Save RL models"""
        import json
        
        # Save DQN
        self.dqn_agent.save(dqn_path)
        print(f"✅ Saved DQN to {dqn_path}")
        
        # Save Bandit
        with open(bandit_path, 'w') as f:
            json.dump(self.bandit.save_state(), f, indent=2)
        print(f"✅ Saved Bandit to {bandit_path}")
    
    def load_models(self, dqn_path: str, bandit_path: str):
        """Load RL models"""
        import json
        
        # Load DQN
        self.dqn_agent.load(dqn_path)
        print(f"✅ Loaded DQN from {dqn_path}")
        
        # Load Bandit
        with open(bandit_path, 'r') as f:
            bandit_state = json.load(f)
        self.bandit.load_state(bandit_state)
        print(f"✅ Loaded Bandit from {bandit_path}")