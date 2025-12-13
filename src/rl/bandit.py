# src/rl/bandit.py
"""
Contextual Bandits with Thompson Sampling
For intelligent source selection and exploration
"""

import numpy as np
from typing import Dict, Any, List
from scipy.stats import beta

class ContextualBandit:
    """
    Contextual Bandit for source selection using Thompson Sampling
    
    Each "arm" represents a different research strategy or source type:
    - Arm 0: Top relevance papers (exploit)
    - Arm 1: High credibility papers (quality focus)
    - Arm 2: Recent papers (recency focus)
    - Arm 3: Diverse domains (exploration)
    - Arm 4: High citation papers (impact focus)
    """
    
    def __init__(self, n_arms: int = 5, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize contextual bandit
        
        Args:
            n_arms: Number of arms (strategies)
            alpha_prior: Prior alpha for Beta distribution (successes)
            beta_prior: Prior beta for Beta distribution (failures)
        """
        self.n_arms = n_arms
        
        # Beta distribution parameters for each arm
        # Start with uniform prior: Beta(1, 1)
        self.alphas = np.ones(n_arms) * alpha_prior
        self.betas = np.ones(n_arms) * beta_prior
        
        # Track arm statistics
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.arm_names = [
            "Top Relevance",
            "High Credibility", 
            "Recent Papers",
            "Diverse Domains",
            "High Citation"
        ]
        
        # History
        self.selection_history = []
        self.reward_history = []
    
    def select_arm(self, context: Dict[str, Any] = None) -> int:
        """
        Select arm using Thompson Sampling
        
        Args:
            context: Optional context features (query type, domain, etc.)
        
        Returns:
            Selected arm index
        """
        # Sample from Beta distribution for each arm
        samples = []
        for i in range(self.n_arms):
            # Thompson Sampling: sample from posterior
            sample = beta.rvs(self.alphas[i], self.betas[i])
            samples.append(sample)
        
        # Select arm with highest sample
        selected_arm = np.argmax(samples)
        
        # Track selection
        self.selection_history.append(selected_arm)
        self.arm_counts[selected_arm] += 1
        
        return selected_arm
    
    def update(self, arm: int, reward: float):
        """
        Update arm statistics based on reward
        
        Args:
            arm: Arm that was selected
            reward: Reward received (0 to 1)
        """
        # Convert reward to binary outcome
        # Reward > 0.6 is "success", else "failure"
        success = reward > 0.6
        
        if success:
            self.alphas[arm] += 1  # Increment successes
        else:
            self.betas[arm] += 1   # Increment failures
        
        # Track rewards
        self.arm_rewards[arm] += reward
        self.reward_history.append((arm, reward))
    
    def get_arm_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for each arm"""
        stats = []
        
        for i in range(self.n_arms):
            # Calculate expected value (mean of Beta distribution)
            expected_value = self.alphas[i] / (self.alphas[i] + self.betas[i])
            
            # Calculate uncertainty (variance)
            variance = (self.alphas[i] * self.betas[i]) / \
                      ((self.alphas[i] + self.betas[i])**2 * 
                       (self.alphas[i] + self.betas[i] + 1))
            
            stats.append({
                'arm': i,
                'name': self.arm_names[i],
                'expected_value': expected_value,
                'uncertainty': np.sqrt(variance),
                'total_pulls': int(self.arm_counts[i]),
                'total_reward': self.arm_rewards[i],
                'avg_reward': self.arm_rewards[i] / max(1, self.arm_counts[i]),
                'alpha': self.alphas[i],
                'beta': self.betas[i]
            })
        
        return stats
    
    def get_best_arm(self) -> int:
        """Get arm with highest expected value"""
        expected_values = self.alphas / (self.alphas + self.betas)
        return int(np.argmax(expected_values))
    
    def get_exploration_rate(self) -> float:
        """Calculate current exploration rate"""
        if len(self.selection_history) < 10:
            return 1.0
        
        # Look at last 20 selections
        recent = self.selection_history[-20:]
        unique_arms = len(set(recent))
        
        # Exploration rate = diversity of recent selections
        return unique_arms / self.n_arms
    
    def reset(self):
        """Reset bandit to initial state"""
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
        self.arm_counts = np.zeros(self.n_arms)
        self.arm_rewards = np.zeros(self.n_arms)
        self.selection_history = []
        self.reward_history = []
    
    def save_state(self) -> Dict[str, Any]:
        """Save bandit state"""
        return {
        'alphas': [float(x) for x in self.alphas],  # Convert to float
        'betas': [float(x) for x in self.betas],    # Convert to float
        'arm_counts': [float(x) for x in self.arm_counts],  # Convert to float
        'arm_rewards': [float(x) for x in self.arm_rewards],  # Convert to float
        'selection_history': [int(x) for x in self.selection_history],  # Convert to int
        'reward_history': [(int(arm), float(reward)) for arm, reward in self.reward_history]  # Convert tuples
    }
    
    def load_state(self, state: Dict[str, Any]):
        """Load bandit state"""
        self.alphas = np.array(state['alphas'])
        self.betas = np.array(state['betas'])
        self.arm_counts = np.array(state['arm_counts'])
        self.arm_rewards = np.array(state['arm_rewards'])
        self.selection_history = state['selection_history']
        self.reward_history = state['reward_history']


class SourceSelectionBandit(ContextualBandit):
    """Specialized bandit for research paper source selection"""
    
    def apply_strategy(self, arm: int, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply selected arm strategy to paper ranking
        
        Args:
            arm: Selected strategy
            papers: List of papers with scores
            
        Returns:
            Re-ranked papers according to strategy
        """
        papers_copy = papers.copy()
        
        if arm == 0:  # Top Relevance
            papers_copy.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        elif arm == 1:  # High Credibility
            papers_copy.sort(
                key=lambda x: x.get('credibility', {}).get('credibility_score', 0), 
                reverse=True
            )
        
        elif arm == 2:  # Recent Papers
            papers_copy.sort(
                key=lambda x: x.get('metadata', {}).get('published', ''), 
                reverse=True
            )
        
        elif arm == 3:  # Diverse Domains
            # Ensure diversity: pick top paper from each domain
            domains_seen = set()
            diverse_papers = []
            
            for paper in papers_copy:
                domain = paper.get('metadata', {}).get('domain', 'unknown')
                if domain not in domains_seen:
                    diverse_papers.append(paper)
                    domains_seen.add(domain)
            
            # Add remaining papers
            for paper in papers_copy:
                if paper not in diverse_papers:
                    diverse_papers.append(paper)
            
            papers_copy = diverse_papers
        
        elif arm == 4:  # High Citation (using relevance as proxy)
            # Sort by combined score
            papers_copy.sort(
                key=lambda x: x.get('combined_score', x.get('score', 0)), 
                reverse=True
            )
        
        return papers_copy