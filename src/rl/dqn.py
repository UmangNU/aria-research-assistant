# src/rl/dqn.py
"""
Deep Q-Network (DQN) for ARIA Research Assistant
Learns optimal agent action sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List
import random

class DQN(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize DQN
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_dims: Hidden layer dimensions
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)


class DQNAgent:
    """DQN Agent for learning research strategies"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 target_update_freq: int = 10):
        """
        Initialize DQN Agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            batch_size: Batch size for training
            target_update_freq: How often to update target network
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-Networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training stats
        self.training_step = 0
        self.losses = []
        self.q_values = []
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            explore: Whether to explore (True) or exploit (False)
            
        Returns:
            Selected action
        """
        if explore and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                self.q_values.append(q_values.max().item())
                return q_values.argmax().item()
    
    def train_step(self, replay_buffer) -> float:
        """
        Perform one training step
        
        Args:
            replay_buffer: Replay buffer with transitions
            
        Returns:
            Loss value
        """
        if not replay_buffer.is_ready(self.batch_size):
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Track loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0.0
        }