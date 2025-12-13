# src/rl/maml.py
"""
Model-Agnostic Meta-Learning (MAML) for ARIA
Enables rapid adaptation to new research domains
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
import copy

class MAMLNetwork(nn.Module):
    """Meta-learnable network for domain adaptation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(MAMLNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class MAMLAgent:
    """
    MAML Agent for fast domain adaptation
    Fourth RL method - demonstrates meta-learning
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 meta_lr: float = 0.001,
                 inner_lr: float = 0.01,
                 num_inner_steps: int = 5):
        """
        Initialize MAML Agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            meta_lr: Meta-learning rate (outer loop)
            inner_lr: Task-specific learning rate (inner loop)
            num_inner_steps: Number of gradient steps for adaptation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        # Meta-model
        self.meta_model = MAMLNetwork(state_dim, action_dim)
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=meta_lr)
        
        # Domain-specific models (cached)
        self.domain_models = {}
        
        # Stats
        self.meta_losses = []
        self.adaptation_losses = []
        self.training_step = 0
    
    def adapt_to_domain(self, domain: str, support_data: List[Tuple]) -> nn.Module:
        """
        Quickly adapt to new domain using few-shot examples
        
        Args:
            domain: Domain name (e.g., 'cs_ml', 'biology')
            support_data: List of (state, action, reward) tuples for adaptation
        
        Returns:
            Adapted model for this domain
        """
        # Clone meta-model
        adapted_model = copy.deepcopy(self.meta_model)
        
        if len(support_data) == 0:
            return adapted_model
        
        # Inner loop: adapt to domain
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.num_inner_steps):
            total_loss = 0
            
            for state, action, reward in support_data:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_tensor = torch.LongTensor([action])
                reward_tensor = torch.FloatTensor([reward])
                
                # Predict Q-values
                q_values = adapted_model(state_tensor)
                
                # Loss: difference between predicted Q and reward
                predicted_q = q_values[0, action_tensor]
                loss = nn.MSELoss()(predicted_q, reward_tensor)
                
                total_loss += loss
            
            # Update adapted model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            self.adaptation_losses.append(total_loss.item())
        
        # Cache adapted model
        self.domain_models[domain] = adapted_model
        
        return adapted_model
    
    def select_action(self, state: np.ndarray, domain: str = 'general') -> int:
        """
        Select action using domain-adapted model
        
        Args:
            state: Current state
            domain: Research domain
        
        Returns:
            Selected action
        """
        # Use domain-specific model if available, else meta-model
        model = self.domain_models.get(domain, self.meta_model)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def meta_update(self, task_batch: List[Dict[str, Any]]):
        """
        Meta-learning update across multiple domains/tasks
        
        Args:
            task_batch: List of tasks, each with support and query sets
        """
        meta_loss = 0
        
        for task in task_batch:
            support_set = task['support']  # Few-shot examples
            query_set = task['query']      # Test examples
            domain = task['domain']
            
            # Adapt to task
            adapted_model = self.adapt_to_domain(domain, support_set)
            
            # Evaluate on query set
            task_loss = 0
            for state, action, reward in query_set:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_tensor = torch.LongTensor([action])
                reward_tensor = torch.FloatTensor([reward])
                
                q_values = adapted_model(state_tensor)
                predicted_q = q_values[0, action_tensor]
                loss = nn.MSELoss()(predicted_q, reward_tensor)
                
                task_loss += loss
            
            meta_loss += task_loss
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), 0.5)
        self.meta_optimizer.step()
        
        self.meta_losses.append(meta_loss.item())
        self.training_step += 1
        
        return meta_loss.item()
    
    def save(self, path: str):
        """Save meta-model"""
        torch.save({
            'meta_model': self.meta_model.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
    
    def load(self, path: str):
        """Load meta-model"""
        checkpoint = torch.load(path)
        self.meta_model.load_state_dict(checkpoint['meta_model'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.training_step = checkpoint['training_step']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'training_steps': self.training_step,
            'avg_meta_loss': np.mean(self.meta_losses[-50:]) if self.meta_losses else 0.0,
            'domains_adapted': len(self.domain_models),
            'adaptation_loss': np.mean(self.adaptation_losses[-50:]) if self.adaptation_losses else 0.0
        }