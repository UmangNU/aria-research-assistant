# src/rl/replay_buffer.py
"""
Experience Replay Buffer for DQN
Stores and samples transitions for training
"""

import numpy as np
from typing import Tuple, List
import random
from collections import deque

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, 
            state: np.ndarray, 
            action: int, 
            reward: float, 
            next_state: np.ndarray, 
            done: bool):
        """
        Add transition to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample random batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples"""
        return len(self.buffer) >= batch_size