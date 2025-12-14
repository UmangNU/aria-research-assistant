# src/agents/adaptive_summarizer.py
"""Adaptive summarization for different user expertise levels"""

from src.agents.base_agent import BaseAgent
from src.utils.llm import get_llm
from typing import Dict, Any

class AdaptiveSummarizer(BaseAgent):
    """Adapts output to user expertise level"""
    
    def __init__(self):
        super().__init__(name="Adaptive Summarizer", role="User-level adapted summaries")
        self.llm = get_llm()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate expertise-adapted summary
        
        Input: {
            'summary': str (base summary),
            'query': str,
            'user_level': str (undergraduate/phd/industry/public)
        }
        """
        
        base = input_data.get('summary', '')
        query = input_data.get('query', '')
        level = input_data.get('user_level', 'phd')
        
        adaptations = {
            'undergraduate': 'Simple language, define all terms, include examples',
            'phd': 'Technical depth, methodology focus, theoretical foundations',
            'industry': 'Applications focus, practical implications, business value',
            'public': 'Accessible language, no jargon, real-world analogies'
        }
        
        prompt = f"""Adapt this research summary for a {level} audience.

Original: {base[:1000]}

Requirements: {adaptations[level]}

Rewrite for {level} level:"""
        
        adapted = self.llm.generate(prompt, max_tokens=800)
        
        return {'adapted_summary': adapted, 'level': level, 'agent': self.name}