# src/utils/error_handler.py
"""
Production-grade error handling for ARIA
Implements retry logic, timeouts, and graceful degradation
"""

import time
import functools
from typing import Callable, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIAException(Exception):
    """Base exception for ARIA system"""
    pass

class AgentExecutionError(ARIAException):
    """Agent execution failed"""
    pass

class LLMError(ARIAException):
    """LLM API error"""
    pass

class VectorStoreError(ARIAException):
    """Vector store error"""
    pass

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= exponential_base
            
        return wrapper
    return decorator

def timeout_handler(timeout_seconds: int = 30):
    """
    Timeout decorator for long-running operations
    
    Args:
        timeout_seconds: Maximum execution time
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal
            
            def timeout_error(signum, frame):
                raise TimeoutError(f"{func.__name__} exceeded {timeout_seconds}s timeout")
            
            # Set alarm (Unix only)
            try:
                signal.signal(signal.SIGALRM, timeout_error)
                signal.alarm(timeout_seconds)
                
                result = func(*args, **kwargs)
                
                signal.alarm(0)  # Cancel alarm
                return result
            except AttributeError:
                # Windows doesn't have SIGALRM - just run normally
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def safe_execute(func: Callable, fallback_value: Any = None, log_error: bool = True) -> Any:
    """
    Safely execute function with fallback
    
    Args:
        func: Function to execute
        fallback_value: Value to return on error
        log_error: Whether to log errors
    
    Returns:
        Function result or fallback value
    """
    try:
        return func()
    except Exception as e:
        if log_error:
            logger.error(f"Error in {func.__name__ if hasattr(func, '__name__') else 'function'}: {e}")
        return fallback_value

class CircuitBreaker:
    """
    Circuit breaker pattern for failing services
    Prevents cascading failures
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker
        
        Args:
            func: Function to call
            *args, **kwargs: Function arguments
        
        Returns:
            Function result
        
        Raises:
            ARIAException: If circuit is open
        """
        if self.state == 'open':
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                logger.info("Circuit breaker entering half-open state")
            else:
                raise ARIAException(f"Circuit breaker OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if in half-open
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")
            
            raise

# Global circuit breakers for different services
llm_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
vector_store_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)