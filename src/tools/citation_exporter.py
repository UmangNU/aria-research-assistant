"""
Citation Export Tool
Generates BibTeX, RIS, and APA citations for papers

REAL UTILITY - Researchers need this!
"""

from typing import Dict, List, Any

class CitationExporter:
    """Export citations in multiple academic formats"""
    
    def __init__(self):
        self.name = "Citation Exporter"
    
    def to_bibtex(self, paper: Dict) -> str:
        """Generate BibTeX citation"""
        
        metadata = paper.get('metadata', paper)
        title = metadata.get('title', 'Unknown Title')
        authors = metadata.get('authors', [])
        year = metadata.get('published', '2024')[:4]
        
        # Format authors for BibTeX
        if isinstance(authors, str):
            author_list = authors
        elif isinstance(authors, list):
            author_list = ' and '.join(authors[:3])  # First 3 authors
            if len(authors) > 3:
                author_list += ' and others'
        else:
            author_list = 'Unknown'
        
        # Generate cite key (first_author + year)
        first_author = author_list.split()[0].lower().replace(',', '') if author_list != 'Unknown' else 'unknown'
        cite_key = f"{first_author}{year}"
        
        # Clean title (remove special chars that break BibTeX)
        clean_title = title.replace('{', '').replace('}', '').replace('\\', '')
        
        bibtex = f"""@article{{{cite_key},
  title={{{clean_title}}},
  author={{{author_list}}},
  journal={{arXiv preprint}},
  year={{{year}}},
  url={{{metadata.get('pdf_url', metadata.get('id', ''))}}}
}}"""
        
        return bibtex
    
    def to_ris(self, paper: Dict) -> str:
        """Generate RIS citation"""
        
        metadata = paper.get('metadata', paper)
        title = metadata.get('title', 'Unknown')
        authors = metadata.get('authors', [])
        year = metadata.get('published', '2024')[:4]
        
        # Format authors for RIS
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(',')]
        
        ris = f"""TY  - JOUR
TI  - {title}
"""
        
        for author in authors[:5]:
            ris += f"AU  - {author}\n"
        
        ris += f"""PY  - {year}
UR  - {metadata.get('pdf_url', metadata.get('id', ''))}
ER  - 
"""
        
        return ris
    
    def to_apa(self, paper: Dict) -> str:
        """Generate APA 7th edition citation"""
        
        metadata = paper.get('metadata', paper)
        title = metadata.get('title', 'Unknown Title')
        authors = metadata.get('authors', [])
        year = metadata.get('published', '2024')[:4]
        
        # Format authors for APA
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(',')]
        
        if len(authors) == 0:
            author_text = "Unknown"
        elif len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} & {authors[1]}"
        else:
            author_text = f"{authors[0]}, {authors[1]}, et al."
        
        apa = f"{author_text}. ({year}). {title}. *arXiv preprint*. {metadata.get('pdf_url', metadata.get('id', ''))}"
        
        return apa
    
    def export_batch(self, papers: List[Dict], format: str = 'bibtex') -> str:
        """
        Export multiple papers in specified format
        
        Args:
            papers: List of paper dicts
            format: 'bibtex', 'ris', or 'apa'
        
        Returns:
            Formatted citations
        """
        
        if format == 'bibtex':
            citations = [self.to_bibtex(p) for p in papers]
            return '\n\n'.join(citations)
        
        elif format == 'ris':
            citations = [self.to_ris(p) for p in papers]
            return '\n'.join(citations)
        
        elif format == 'apa':
            citations = [self.to_apa(p) for p in papers]
            return '\n\n'.join(citations)
        
        else:
            return "Invalid format. Use 'bibtex', 'ris', or 'apa'."
    
    def export_all_formats(self, papers: List[Dict]) -> Dict[str, str]:
        """Export in all formats at once"""
        
        return {
            'bibtex': self.export_batch(papers, 'bibtex'),
            'ris': self.export_batch(papers, 'ris'),
            'apa': self.export_batch(papers, 'apa'),
            'count': len(papers)
        }