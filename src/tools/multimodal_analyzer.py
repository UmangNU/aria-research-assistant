# src/tools/multimodal_analyzer.py
"""
Multi-Modal PDF Analysis Tool
Extracts and analyzes figures, tables, and text from academic PDFs

UNIQUE INNOVATION - Most research tools only use abstracts!
"""

import fitz  # PyMuPDF
from PIL import Image
import io
from typing import Dict, Any, List, Tuple
import os

class MultiModalAnalyzer:
    """Extract and analyze visual elements from PDFs"""
    
    def __init__(self):
        self.name = "Multi-Modal PDF Analyzer"
        
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text, figures, and tables from PDF
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with extracted content
        """
        
        if not os.path.exists(pdf_path):
            return {'error': 'PDF not found', 'figures': [], 'tables': [], 'text': ''}
        
        try:
            doc = fitz.open(pdf_path)
            
            all_text = []
            figures = []
            tables = []
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                all_text.append(text)
                
                # Extract images (figures)
                images = page.get_images()
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    figures.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'width': base_image['width'],
                        'height': base_image['height'],
                        'format': base_image['ext'],
                        'size_bytes': len(base_image['image'])
                    })
                
                # Extract tables (simple heuristic: look for aligned text)
                # More sophisticated: use table detection models
                table_text = self._detect_tables_heuristic(text)
                if table_text:
                    tables.append({
                        'page': page_num + 1,
                        'content': table_text
                    })
            
            doc.close()
            
            return {
                'text': '\n\n'.join(all_text),
                'figures': figures,
                'tables': tables,
                'pages': len(doc),
                'has_figures': len(figures) > 0,
                'has_tables': len(tables) > 0
            }
            
        except Exception as e:
            return {'error': str(e), 'figures': [], 'tables': [], 'text': ''}
    
    def _detect_tables_heuristic(self, text: str) -> str:
        """Simple heuristic table detection"""
        
        lines = text.split('\n')
        
        # Look for lines with multiple aligned numbers/columns
        table_lines = []
        for line in lines:
            # Simple heuristic: line with 3+ numbers or lots of whitespace
            if line.count(' ') > 10 and any(c.isdigit() for c in line):
                table_lines.append(line)
        
        if len(table_lines) > 3:  # At least 3 rows
            return '\n'.join(table_lines)
        
        return ""
    
    def analyze_visual_content(self, 
                               figures: List[Dict],
                               tables: List[Dict]) -> str:
        """
        Generate description of visual content
        
        Args:
            figures: List of figure metadata
            tables: List of table content
        
        Returns:
            Analysis text
        """
        
        analysis = []
        
        if figures:
            analysis.append(f"**Figures:** Document contains {len(figures)} figures across {len(set(f['page'] for f in figures))} pages.")
            
            # Analyze figure characteristics
            avg_size = sum(f['size_bytes'] for f in figures) / len(figures)
            formats = set(f['format'] for f in figures)
            
            analysis.append(f"Average figure size: {avg_size/1024:.1f} KB. Formats: {', '.join(formats)}.")
        
        if tables:
            analysis.append(f"**Tables:** Document contains {len(tables)} tables.")
            analysis.append("Tables provide quantitative results and comparisons.")
        
        if not figures and not tables:
            analysis.append("Document is text-only (no figures or tables detected).")
        
        return ' '.join(analysis)
    
    def enrich_paper_metadata(self, paper: Dict[str, Any], pdf_path: str = None) -> Dict[str, Any]:
        """
        Enrich paper metadata with multi-modal analysis
        
        Args:
            paper: Paper dictionary
            pdf_path: Path to PDF (optional)
        
        Returns:
            Enhanced paper dict with visual analysis
        """
        
        if not pdf_path:
            # Can't analyze without PDF
            paper['multimodal_analysis'] = {'available': False, 'reason': 'No PDF provided'}
            return paper
        
        # Extract from PDF
        extracted = self.extract_from_pdf(pdf_path)
        
        # Analyze visual content
        if 'error' not in extracted:
            visual_analysis = self.analyze_visual_content(extracted['figures'], extracted['tables'])
            
            paper['multimodal_analysis'] = {
                'available': True,
                'figures_count': len(extracted['figures']),
                'tables_count': len(extracted['tables']),
                'pages': extracted['pages'],
                'has_visuals': extracted['has_figures'] or extracted['has_tables'],
                'visual_summary': visual_analysis
            }
        else:
            paper['multimodal_analysis'] = {
                'available': False,
                'error': extracted['error']
            }
        
        return paper