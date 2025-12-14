# src/agents/query_analyzer.py
from src.agents.base_agent import BaseAgent
from typing import Dict, Any

class QueryAnalyzerAgent(BaseAgent):
    """Analyzes and classifies research queries"""
    
    def __init__(self):
        super().__init__(
            name="Query Analyzer",
            role="Understanding research questions and extracting key concepts"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query and extract metadata
        
        Input: {'query': str}
        Output: {
            'query': str,
            'keywords': List[str],
            'domain': str,
            'complexity': str,
            'query_type': str
        }
        """
        query = input_data.get('query', '')
        
        # Extract keywords (simple approach - split and filter)
        words = query.lower().split()
        stopwords = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an', 'in', 'for', 'to', 'of'}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Classify domain (simple keyword matching)
        domain = self._classify_domain(query)
        
        # Assess complexity
        complexity = self._assess_complexity(query)
        
        # Determine query type
        query_type = self._classify_query_type(query)
        
        result = {
            'query': query,
            'keywords': keywords[:5],  # Top 5 keywords
            'domain': domain,
            'complexity': complexity,
            'query_type': query_type,
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _classify_domain(self, query: str) -> str:
        """Classify research domain"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['machine learning', 'deep learning', 'neural network', 'ai', 'reinforcement']):
            return 'cs_ml'
        elif any(term in query_lower for term in ['nlp', 'language', 'text', 'translation', 'sentiment']):
            return 'cs_nlp'
        elif any(term in query_lower for term in ['vision', 'image', 'detection', 'segmentation', 'visual']):
            return 'cs_cv'
        elif any(term in query_lower for term in ['protein', 'gene', 'dna', 'cell', 'biology', 'molecular']):
            return 'biology'
        elif any(term in query_lower for term in ['physics', 'quantum', 'particle', 'energy']):
            return 'physics'
        elif any(term in query_lower for term in ['medical', 'disease', 'clinical', 'patient', 'treatment']):
            return 'medicine'
        else:
            return 'general'
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        
        if word_count < 5:
            return 'simple'
        elif word_count < 15:
            return 'moderate'
        else:
            return 'complex'
    
    def _classify_query_type(self, query: str) -> str:
        """Classify type of research query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['recent', 'latest', 'advances', 'progress', 'state of the art']):
            return 'literature_review'
        elif any(term in query_lower for term in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(term in query_lower for term in ['how', 'method', 'approach', 'technique']):
            return 'methodology'
        elif any(term in query_lower for term in ['what is', 'define', 'definition']):
            return 'definition'
        else:
            return 'exploratory'