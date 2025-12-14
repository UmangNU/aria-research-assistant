"""
Research Timeline Visualization
Shows how research evolved over time

VISUAL WOW FACTOR!
"""

from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from collections import defaultdict

class TimelineVisualizer:
    """Visualize research evolution over time"""
    
    def __init__(self):
        self.name = "Research Timeline Visualizer"
    
    def create_timeline(self, papers: List[Dict]) -> Dict[str, Any]:
        """
        Create timeline from papers
        
        Args:
            papers: List of analyzed papers
        
        Returns:
            Timeline data and visualization
        """
        
        if not papers:
            return {'error': 'No papers provided'}
        
        # Extract temporal data
        timeline_data = self._extract_timeline_data(papers)
        
        # Create visualization (save to file)
        fig_path = self._plot_timeline(timeline_data)
        
        return {
            'timeline_data': timeline_data,
            'figure_path': fig_path,
            'total_papers': len(papers),
            'year_range': timeline_data['year_range'],
            'key_periods': timeline_data['key_periods']
        }
    
    def _extract_timeline_data(self, papers: List[Dict]) -> Dict:
        """Extract and structure timeline data"""
        
        # Group papers by year
        papers_by_year = defaultdict(list)
        
        for paper in papers:
            pub_date = paper.get('published', '2024-01-01')
            year = int(pub_date[:4])
            
            papers_by_year[year].append({
                'title': paper.get('title', 'Unknown'),
                'year': year,
                'domain': paper.get('domain', 'general')
            })
        
        # Find key periods (years with most papers)
        sorted_years = sorted(papers_by_year.items(), key=lambda x: len(x[1]), reverse=True)
        key_periods = [
            {
                'year': year,
                'paper_count': len(papers_list),
                'significance': 'High activity' if len(papers_list) > len(papers) * 0.2 else 'Active research'
            }
            for year, papers_list in sorted_years[:5]
        ]
        
        years = sorted(papers_by_year.keys())
        year_range = f"{min(years)}-{max(years)}" if years else "N/A"
        
        return {
            'papers_by_year': dict(papers_by_year),
            'year_counts': {year: len(papers_list) for year, papers_list in papers_by_year.items()},
            'year_range': year_range,
            'key_periods': key_periods,
            'total_years': len(years)
        }
    
    def _plot_timeline(self, timeline_data: Dict) -> str:
        """Create timeline visualization"""
        
        import os
        os.makedirs('temp_figures', exist_ok=True)
        
        year_counts = timeline_data['year_counts']
        
        if not year_counts:
            return None
        
        years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot
        ax.plot(years, counts, 'o-', linewidth=3, markersize=10, color='steelblue')
        ax.fill_between(years, counts, alpha=0.3, color='steelblue')
        
        # Styling
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Papers', fontsize=14, fontweight='bold')
        ax.set_title('Research Timeline: Publication Trends', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight key years
        max_count = max(counts)
        max_year = years[counts.index(max_count)]
        ax.annotate(f'Peak: {max_count} papers',
                   xy=(max_year, max_count),
                   xytext=(max_year, max_count + 1),
                   fontsize=11,
                   fontweight='bold',
                   ha='center',
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        plt.tight_layout()
        
        # Save
        fig_path = 'temp_figures/research_timeline.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path