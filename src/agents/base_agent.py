# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = []
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's task"""
        pass
    
    def add_to_memory(self, data: Dict[str, Any]):
        """Store information in memory"""
        self.memory.append(data)
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Retrieve agent's memory"""
        return self.memory
    
    def __str__(self):
        return f"{self.name} ({self.role})"