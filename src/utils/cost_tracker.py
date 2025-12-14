"""
API Cost Tracking
Real-time cost calculation for OpenAI API usage

PRODUCTION AWARENESS - Shows you understand economics!
"""

from typing import Dict, List, Tuple
from datetime import datetime

class CostTracker:
    """Track API costs in real-time"""
    
    def __init__(self):
        self.name = "API Cost Tracker"
        
        # GPT-4o-mini pricing (Dec 2024)
        self.pricing = {
            'gpt-4o-mini': {
                'input': 0.150 / 1_000_000,   # $0.150 per 1M input tokens
                'output': 0.600 / 1_000_000   # $0.600 per 1M output tokens
            },
            'gpt-4o': {
                'input': 2.50 / 1_000_000,
                'output': 10.00 / 1_000_000
            }
        }
        
        # Session tracking
        self.session_costs = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.query_count = 0
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens (rough: 1 token ≈ 4 chars)"""
        return len(text) // 4
    
    def calculate_cost(self, 
                      input_tokens: int, 
                      output_tokens: int, 
                      model: str = 'gpt-4o-mini') -> Dict:
        """
        Calculate cost for API call
        
        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            model: Model name
        
        Returns:
            Cost breakdown
        """
        
        if model not in self.pricing:
            model = 'gpt-4o-mini'  # Default
        
        input_cost = input_tokens * self.pricing[model]['input']
        output_cost = output_tokens * self.pricing[model]['output']
        total_cost = input_cost + output_cost
        
        # Track session
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        
        self.session_costs.append({
            'timestamp': datetime.now().isoformat(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': total_cost,
            'model': model
        })
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'total_cost': round(total_cost, 6),
            'model': model
        }
    
    def track_query(self, prompt: str, response: str, model: str = 'gpt-4o-mini'):
        """
        Track a complete query (estimate tokens)
        
        Args:
            prompt: Input prompt
            response: Generated response
            model: Model used
        """
        
        input_tokens = self.estimate_tokens(prompt)
        output_tokens = self.estimate_tokens(response)
        
        self.calculate_cost(input_tokens, output_tokens, model)
        self.query_count += 1
    
    def get_session_summary(self) -> Dict:
        """Get session cost summary"""
        
        return {
            'total_queries': self.query_count,
            'total_api_calls': len(self.session_costs),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_cost': round(self.total_cost, 4),
            'avg_cost_per_query': round(self.total_cost / max(self.query_count, 1), 4),
            'avg_tokens_per_query': round((self.total_input_tokens + self.total_output_tokens) / max(self.query_count, 1), 1),
            'cost_breakdown': {
                'input_cost': round(self.total_input_tokens * self.pricing['gpt-4o-mini']['input'], 4),
                'output_cost': round(self.total_output_tokens * self.pricing['gpt-4o-mini']['output'], 4)
            }
        }
    
    def format_cost(self, cost: float) -> str:
        """Format cost for display"""
        
        if cost < 0.001:
            return f"${cost:.6f}"
        elif cost < 0.01:
            return f"${cost:.4f}"
        elif cost < 1.00:
            return f"${cost:.3f}"
        else:
            return f"${cost:.2f}"
    
    def get_cost_metrics(self) -> Dict[str, str]:
        """Get formatted cost metrics for display"""
        
        summary = self.get_session_summary()
        
        return {
            'total_cost': self.format_cost(summary['total_cost']),
            'avg_per_query': self.format_cost(summary['avg_cost_per_query']),
            'total_tokens': f"{summary['total_tokens']:,}",
            'queries': str(summary['total_queries'])
        }
    
    def reset_session(self):
        """Reset session tracking"""
        self.session_costs = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.query_count = 0