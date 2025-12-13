# src/utils/logger.py
"""
Comprehensive logging system for ARIA
Tracks all agent actions, RL decisions, and system events
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any
import json

class ARIALogger:
    """Custom logger for ARIA system"""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize logging system"""
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"aria_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.logger = logging.getLogger('ARIA')
        self.events = []
        self.log_file = log_file
        
        self.logger.info(f"ARIA Logger initialized - logs: {log_file}")
    
    def log_research_start(self, query: str, config: Dict[str, Any]):
        """Log research query start"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'research_start',
            'query': query,
            'config': config
        }
        self.events.append(event)
        self.logger.info(f"Research started: {query[:50]}...")
    
    def log_agent_execution(self, agent_name: str, input_data: Dict, output_data: Dict):
        """Log agent execution"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'agent_execution',
            'agent': agent_name,
            'input_size': len(str(input_data)),
            'output_size': len(str(output_data))
        }
        self.events.append(event)
        self.logger.debug(f"Agent {agent_name} executed")
    
    def log_rl_decision(self, method: str, state_dim: int, action: int, reward: float):
        """Log RL decision"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'rl_decision',
            'method': method,
            'state_dim': state_dim,
            'action': action,
            'reward': reward
        }
        self.events.append(event)
        self.logger.info(f"{method} selected action {action}, reward: {reward:.3f}")
    
    def log_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'error',
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        self.events.append(event)
        self.logger.error(f"Error in {component}: {error}")
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'performance_metrics',
            'metrics': metrics
        }
        self.events.append(event)
        self.logger.info(f"Metrics: {metrics}")
    
    def save_session_log(self, output_path: str = None):
        """Save complete session log"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/session_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.events, f, indent=2)
        
        self.logger.info(f"Session log saved to: {output_path}")
        return output_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        from collections import Counter
        
        event_types = Counter(e['event'] for e in self.events)
        
        errors = [e for e in self.events if e['event'] == 'error']
        error_types = Counter(e['error_type'] for e in errors)
        
        return {
            'total_events': len(self.events),
            'event_types': dict(event_types),
            'total_errors': len(errors),
            'error_types': dict(error_types),
            'session_duration': self._calculate_duration()
        }
    
    def _calculate_duration(self) -> float:
        """Calculate session duration in seconds"""
        if len(self.events) < 2:
            return 0.0
        
        start = datetime.fromisoformat(self.events[0]['timestamp'])
        end = datetime.fromisoformat(self.events[-1]['timestamp'])
        
        return (end - start).total_seconds()

# Global logger instance
_aria_logger = None

def get_logger():
    """Get or create ARIA logger singleton"""
    global _aria_logger
    if _aria_logger is None:
        _aria_logger = ARIALogger()
    return _aria_logger