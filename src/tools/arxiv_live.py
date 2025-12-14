# src/tools/arxiv_live.py
"""Real-time arXiv integration - always current papers"""

import arxiv
from datetime import datetime, timedelta
from typing import List, Dict

class ArxivLiveIntegration:
    """Fetch newest papers from arXiv in real-time"""
    
    def __init__(self):
        self.name = "Real-time arXiv Integration"
    
    def get_latest_papers(self, query: str, days_back: int = 7, max_results: int = 10) -> List[Dict]:
        """Get papers from last N days"""
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_papers = []
        
        for paper in search.results():
            if paper.published.replace(tzinfo=None) > cutoff_date:
                recent_papers.append({
                    'id': paper.entry_id,
                    'title': paper.title,
                    'abstract': paper.summary,
                    'authors': [a.name for a in paper.authors],
                    'published': paper.published.isoformat(),
                    'is_new': True,
                    'days_old': (datetime.now() - paper.published.replace(tzinfo=None)).days
                })
        
        return recent_papers