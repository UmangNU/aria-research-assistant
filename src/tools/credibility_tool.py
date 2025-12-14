# src/tools/credibility_tool.py
"""
Custom Tool: Academic Paper Credibility Scorer
This tool evaluates the credibility and reliability of research papers
"""

from typing import Dict, Any, List
from datetime import datetime

class CredibilityTool:
    """
    Custom tool for assessing research paper credibility
    
    This is a CUSTOM TOOL (not built-in) that provides:
    - Multi-factor credibility assessment
    - Venue quality scoring
    - Recency evaluation
    - Citation quality estimation
    """
    
    def __init__(self):
        self.name = "Academic Credibility Scorer"
        self.version = "1.0.0"
        self.description = "Evaluates credibility of academic papers based on multiple factors"
        
        # High-impact venues
        self.top_venues = {
            'Nature': 1.0,
            'Science': 1.0,
            'Cell': 1.0,
            'PNAS': 0.95,
            'NeurIPS': 0.95,
            'ICML': 0.95,
            'ICLR': 0.95,
            'CVPR': 0.9,
            'ACL': 0.9,
            'EMNLP': 0.9,
            'AAAI': 0.85,
            'IJCAI': 0.85
        }
        
        # Category weights
        self.category_weights = {
            'cs.LG': 1.0,
            'cs.AI': 1.0,
            'cs.CL': 0.95,
            'cs.CV': 0.95,
            'stat.ML': 0.95,
            'q-bio': 0.9,
            'physics': 0.85,
            'math': 0.9
        }
        
        # Quality indicators in titles
        self.quality_indicators = {
            'positive': ['novel', 'state-of-the-art', 'breakthrough', 'efficient', 'improved'],
            'negative': ['preliminary', 'work in progress', 'draft']
        }
    
    def score_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive credibility score for a paper
        
        Args:
            paper: Dictionary containing paper metadata
        
        Returns:
            Dictionary with credibility score and breakdown
        """
        metadata = paper.get('metadata', paper)
        
        # Calculate component scores
        venue_score = self._score_venue(metadata)
        recency_score = self._score_recency(metadata)
        category_score = self._score_category(metadata)
        author_score = self._score_authors(metadata)
        title_score = self._score_title(metadata)
        
        # Weighted combination
        total_score = (
            0.30 * venue_score +
            0.20 * recency_score +
            0.20 * category_score +
            0.15 * author_score +
            0.15 * title_score
        )
        
        return {
            'credibility_score': round(total_score, 3),
            'breakdown': {
                'venue_quality': round(venue_score, 3),
                'recency': round(recency_score, 3),
                'category_relevance': round(category_score, 3),
                'author_count': round(author_score, 3),
                'title_quality': round(title_score, 3)
            },
            'assessment': self._get_assessment(total_score),
            'tool': self.name
        }
    
    def score_papers_batch(self, papers: List[Dict]) -> List[Dict]:
        """Score multiple papers at once"""
        return [self.score_paper(paper) for paper in papers]
    
    def _score_venue(self, metadata: Dict) -> float:
        """Score based on publication venue"""
        title = metadata.get('title', '').lower()
        
        # Check for known venues
        for venue, score in self.top_venues.items():
            if venue.lower() in title:
                return score
        
        # Check for conference/journal indicators
        if any(term in title for term in ['conference', 'proceedings', 'journal']):
            return 0.7
        
        # Default for arXiv preprints
        return 0.6
    
    def _score_recency(self, metadata: Dict) -> float:
        """Score based on publication date"""
        published = metadata.get('published', '')
        
        try:
            if isinstance(published, str):
                pub_year = int(published[:4])
            else:
                return 0.5
            
            current_year = datetime.now().year
            age = current_year - pub_year
            
            # Decay function
            if age == 0:
                return 1.0
            elif age == 1:
                return 0.9
            elif age == 2:
                return 0.75
            elif age == 3:
                return 0.6
            elif age <= 5:
                return 0.4
            else:
                return 0.2
        except:
            return 0.5
    
    def _score_category(self, metadata: Dict) -> float:
        """Score based on research category"""
        categories = metadata.get('categories', [])
        
        if not categories:
            return 0.5
        
        # Get highest scoring category
        max_score = 0.5
        for cat in categories:
            for known_cat, weight in self.category_weights.items():
                if cat.startswith(known_cat):
                    max_score = max(max_score, weight)
        
        return max_score
    
    def _score_authors(self, metadata: Dict) -> float:
        """Score based on author information"""
        authors = metadata.get('authors', '')
        
        if isinstance(authors, str):
            author_count = len(authors.split(','))
        elif isinstance(authors, list):
            author_count = len(authors)
        else:
            return 0.5
        
        # More authors often indicates more rigorous research
        if author_count == 1:
            return 0.5
        elif author_count <= 3:
            return 0.7
        elif author_count <= 6:
            return 0.85
        else:
            return 1.0
    
    def _score_title(self, metadata: Dict) -> float:
        """Score based on title quality indicators"""
        title = metadata.get('title', '').lower()
        
        score = 0.5  # Baseline
        
        # Check for positive indicators
        for indicator in self.quality_indicators['positive']:
            if indicator in title:
                score += 0.1
        
        # Check for negative indicators
        for indicator in self.quality_indicators['negative']:
            if indicator in title:
                score -= 0.15
        
        # Title length (sweet spot: 10-20 words)
        word_count = len(title.split())
        if 10 <= word_count <= 20:
            score += 0.05
        
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _get_assessment(self, score: float) -> str:
        """Get qualitative assessment"""
        if score >= 0.85:
            return "Highly Credible"
        elif score >= 0.70:
            return "Credible"
        elif score >= 0.55:
            return "Moderately Credible"
        elif score >= 0.40:
            return "Low Credibility"
        else:
            return "Very Low Credibility"
    
    def get_tool_info(self) -> Dict[str, str]:
        """Get tool metadata"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'type': 'custom',
            'author': 'ARIA Research Assistant'
        }