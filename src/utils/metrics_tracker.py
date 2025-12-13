# src/utils/metrics_tracker.py
"""
Real-time performance monitoring for ARIA
Tracks latency, throughput, resource usage
"""

import time
import psutil
import os
from typing import Dict, Any, List
from collections import deque
import numpy as np

class PerformanceTracker:
    """Tracks system performance metrics"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance tracker
        
        Args:
            window_size: Number of recent operations to track
        """
        self.window_size = window_size
        
        # Metrics
        self.latencies = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        
        # Agent-specific metrics
        self.agent_latencies = {}
        
        # Process
        self.process = psutil.Process(os.getpid())
        
    def start_operation(self) -> float:
        """Start timing an operation"""
        return time.time()
    
    def end_operation(self, start_time: float, operation_name: str = "general"):
        """
        End timing and record metrics
        
        Args:
            start_time: Start time from start_operation()
            operation_name: Name of operation
        """
        latency = time.time() - start_time
        self.latencies.append(latency)
        
        # Track agent-specific latency
        if operation_name not in self.agent_latencies:
            self.agent_latencies[operation_name] = deque(maxlen=self.window_size)
        self.agent_latencies[operation_name].append(latency)
        
        # System metrics
        self.memory_usage.append(self.process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(self.process.cpu_percent())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        if not self.latencies:
            return {
                'operations': 0,
                'avg_latency': 0.0,
                'avg_memory_mb': 0.0,
                'avg_cpu_percent': 0.0
            }
        
        metrics = {
            'operations': len(self.latencies),
            'avg_latency': np.mean(self.latencies),
            'p50_latency': np.percentile(self.latencies, 50),
            'p95_latency': np.percentile(self.latencies, 95),
            'p99_latency': np.percentile(self.latencies, 99),
            'max_latency': np.max(self.latencies),
            'min_latency': np.min(self.latencies),
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0.0,
            'peak_memory_mb': np.max(self.memory_usage) if self.memory_usage else 0.0,
            'avg_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0.0,
            'agent_latencies': {
                agent: {
                    'avg': np.mean(latencies),
                    'p95': np.percentile(latencies, 95)
                }
                for agent, latencies in self.agent_latencies.items()
                if len(latencies) > 0
            }
        }
        
        return metrics
    
    def get_throughput(self, time_window: float = 60.0) -> float:
        """
        Calculate operations per second
        
        Args:
            time_window: Time window in seconds
        
        Returns:
            Operations per second
        """
        if not self.latencies:
            return 0.0
        
        total_time = sum(self.latencies)
        if total_time == 0:
            return 0.0
        
        return len(self.latencies) / total_time
    
    def print_summary(self):
        """Print performance summary"""
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        print(f"Total Operations: {metrics['operations']}")
        print(f"\nLatency:")
        print(f"  Average: {metrics['avg_latency']*1000:.2f} ms")
        print(f"  P50: {metrics['p50_latency']*1000:.2f} ms")
        print(f"  P95: {metrics['p95_latency']*1000:.2f} ms")
        print(f"  P99: {metrics['p99_latency']*1000:.2f} ms")
        print(f"\nResource Usage:")
        print(f"  Avg Memory: {metrics['avg_memory_mb']:.1f} MB")
        print(f"  Peak Memory: {metrics['peak_memory_mb']:.1f} MB")
        print(f"  Avg CPU: {metrics['avg_cpu_percent']:.1f}%")
        print(f"\nThroughput: {self.get_throughput():.2f} ops/sec")
        
        if metrics['agent_latencies']:
            print(f"\nAgent Latencies:")
            for agent, stats in metrics['agent_latencies'].items():
                print(f"  {agent:20s}: avg={stats['avg']*1000:.2f}ms, p95={stats['p95']*1000:.2f}ms")
        
        print("="*60)

# Global tracker
_perf_tracker = None

def get_performance_tracker():
    """Get or create performance tracker singleton"""
    global _perf_tracker
    if _perf_tracker is None:
        _perf_tracker = PerformanceTracker()
    return _perf_tracker