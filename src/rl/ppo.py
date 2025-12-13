# src/rl/ppo.py
"""
Proximal Policy Optimization (PPO) for ARIA
Learns continuous research strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple

class PPOActor(nn.Module):
    """Policy network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPOActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOCritic(nn.Module):
    """Value network for PPO"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(PPOCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    """
    PPO Agent for learning research strategies
    Third RL method for exceptional performance
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr_actor: float = 0.0003,
                 lr_critic: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_clip: float = 0.2,
                 k_epochs: int = 4):
        """
        Initialize PPO Agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            epsilon_clip: PPO clipping parameter
            k_epochs: Number of epochs per update
        """
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        
        # Networks
        self.actor = PPOActor(state_dim, action_dim)
        self.critic = PPOCritic(state_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        
        # Stats
        self.actor_losses = []
        self.critic_losses = []
        self.training_step = 0
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Select action using current policy
        
        Args:
            state: Current state
        
        Returns:
            (action, log_prob)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def store_transition(self, state, action, reward, is_terminal, log_prob):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.log_probs.append(log_prob)
    
    def train(self):
        """Update policy using PPO"""
        
        if len(self.states) == 0:
            return 0.0, 0.0
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # Calculate discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # Normalize
        
        # PPO update for k epochs
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.k_epochs):
            # Get current predictions
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            state_values = self.critic(states).squeeze()
            
            # Calculate advantages
            advantages = rewards - state_values.detach()
            
            # Actor loss (PPO clipped objective)
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(state_values, rewards)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        
        # Track stats
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        
        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        self.training_step += 1
        
        return avg_actor_loss, avg_critic_loss
    
    def save(self, path: str):
        """Save models"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
    
    def load(self, path: str):
        """Load models"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'training_steps': self.training_step,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0
        }