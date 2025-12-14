# src/rl/__init__.py
"""
Reinforcement Learning Module for ARIA

Implements FOUR RL approaches for exceptional performance:
1. DQN: Deep Q-Network for action selection
2. Contextual Bandits: Source selection with Thompson Sampling
3. PPO: Proximal Policy Optimization for continuous strategies
4. MAML: Model-Agnostic Meta-Learning for domain adaptation
"""

from .dqn import DQN, DQNAgent
from .bandit import ContextualBandit, SourceSelectionBandit
from .ppo import PPOAgent, PPOActor, PPOCritic
from .maml import MAMLAgent, MAMLNetwork
from .state_builder import StateBuilder
from .replay_buffer import ReplayBuffer

__all__ = [
    'DQN',
    'DQNAgent',
    'ContextualBandit',
    'SourceSelectionBandit',
    'PPOAgent',
    'PPOActor',
    'PPOCritic',
    'MAMLAgent',
    'MAMLNetwork',
    'StateBuilder',
    'ReplayBuffer'
]

__version__ = '2.0.0'  # Updated to v2 with 4 RL methods