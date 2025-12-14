# src/utils/validators.py
"""
Input validation and quality checks for ARIA
Ensures data quality and catches errors early
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Validation failed"""
    pass

def validate_query(query: str) -> str:
    """
    Validate and sanitize research query
    
    Args:
        query: User research query
    
    Returns:
        Cleaned query
    
    Raises:
        ValidationError: If query is invalid
    """
    if not query or not isinstance(query, str):
        raise ValidationError("Query must be a non-empty string")
    
    query = query.strip()
    
    if len(query) < 3:
        raise ValidationError("Query too short (minimum 3 characters)")
    
    if len(query) > 500:
        logger.warning(f"Query very long ({len(query)} chars) - truncating to 500")
        query = query[:500]
    
    # Remove problematic characters
    query = query.replace('\n', ' ').replace('\r', '')
    
    return query

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate research configuration
    
    Args:
        config: Research configuration dict
    
    Returns:
        Validated config with defaults
    """
    validated = {
        'top_k': config.get('top_k', 10),
        'depth': config.get('depth', 'moderate'),
        'style': config.get('style', 'detailed')
    }
    
    # Validate ranges
    if validated['top_k'] < 1:
        logger.warning(f"top_k={validated['top_k']} invalid, setting to 1")
        validated['top_k'] = 1
    elif validated['top_k'] > 20:
        logger.warning(f"top_k={validated['top_k']} too high, capping at 20")
        validated['top_k'] = 20
    
    # Validate depth
    valid_depths = ['shallow', 'moderate', 'deep']
    if validated['depth'] not in valid_depths:
        logger.warning(f"depth={validated['depth']} invalid, using 'moderate'")
        validated['depth'] = 'moderate'
    
    # Validate style
    valid_styles = ['concise', 'detailed', 'technical']
    if validated['style'] not in valid_styles:
        logger.warning(f"style={validated['style']} invalid, using 'detailed'")
        validated['style'] = 'detailed'
    
    return validated

def validate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate paper data structure
    
    Args:
        papers: List of paper dictionaries
    
    Returns:
        Validated papers
    """
    if not papers:
        logger.warning("Empty papers list")
        return []
    
    validated = []
    
    for i, paper in enumerate(papers):
        # Check required fields
        if 'metadata' not in paper:
            logger.warning(f"Paper {i} missing metadata, skipping")
            continue
        
        metadata = paper['metadata']
        required_fields = ['title', 'domain', 'published']
        
        if all(field in metadata for field in required_fields):
            validated.append(paper)
        else:
            logger.warning(f"Paper {i} missing required fields")
    
    return validated

def validate_state(state: Any) -> bool:
    """
    Validate RL state vector
    
    Args:
        state: State vector (numpy array)
    
    Returns:
        True if valid, False otherwise
    """
    import numpy as np
    
    if not isinstance(state, np.ndarray):
        logger.error("State must be numpy array")
        return False
    
    if state.ndim != 1:
        logger.error(f"State must be 1D, got {state.ndim}D")
        return False
    
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        logger.error("State contains NaN or Inf values")
        return False
    
    return True

def validate_reward(reward: float) -> float:
    """
    Validate and clip reward value
    
    Args:
        reward: Reward value
    
    Returns:
        Validated reward in valid range
    """
    import numpy as np
    
    if np.isnan(reward) or np.isinf(reward):
        logger.warning(f"Invalid reward {reward}, using 0.0")
        return 0.0
    
    # Clip to reasonable range
    clipped = np.clip(reward, -10.0, 10.0)
    
    if clipped != reward:
        logger.warning(f"Reward {reward} clipped to {clipped}")
    
    return float(clipped)