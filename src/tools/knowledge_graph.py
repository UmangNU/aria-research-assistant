# src/tools/knowledge_graph.py
"""
Knowledge Graph Construction
Builds conceptual relationships between papers, concepts, and authors
"""

from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import json

class KnowledgeGraph:
    """Build and query knowledge graph from papers"""
    
    def __init__(self):
        self.name = "Knowledge Graph Builder"
        self.nodes = {
            'papers': {},      # paper_id -> paper_data
            'concepts': {},    # concept -> papers mentioning it
            'authors': {},     # author -> papers authored
            'methods': {}      # method -> papers using it
        }
        self.edges = {
            'cites': [],           # (paper1, paper2)
            'builds_on': [],       # (paper1, concept)
            'authored_by': [],     # (paper, author)
            'uses_method': [],     # (paper, method)
            'related_to': []       # (concept1, concept2)
        }
    
    def build_from_papers(self, papers: List[Dict[str, Any]]):
        """
        Construct knowledge graph from paper collection
        
        Args:
            papers: List of paper dictionaries
        """
        
        print(f"\n🕸️  Building knowledge graph from {len(papers)} papers...")
        
        for paper in papers:
            paper_id = paper.get('id', paper.get('metadata', {}).get('title', ''))
            metadata = paper.get('metadata', paper)
            
            # Add paper node
            self.nodes['papers'][paper_id] = {
                'title': metadata.get('title', ''),
                'domain': metadata.get('domain', ''),
                'year': metadata.get('published', '')[:4],
                'authors': metadata.get('authors', [])
            }
            
            # Extract and add concepts from title
            title = metadata.get('title', '').lower()
            concepts = self._extract_concepts(title)
            
            for concept in concepts:
                if concept not in self.nodes['concepts']:
                    self.nodes['concepts'][concept] = []
                self.nodes['concepts'][concept].append(paper_id)
                self.edges['builds_on'].append((paper_id, concept))
            
            # Add authors
            authors = metadata.get('authors', [])
            if isinstance(authors, str):
                authors = authors.split(',')
            
            for author in authors[:3]:  # Limit to first 3 authors
                author = author.strip()
                if author:
                    if author not in self.nodes['authors']:
                        self.nodes['authors'][author] = []
                    self.nodes['authors'][author].append(paper_id)
                    self.edges['authored_by'].append((paper_id, author))
        
        # Build concept relationships
        self._build_concept_relationships()
        
        print(f"   ✓ Graph built:")
        print(f"     Papers: {len(self.nodes['papers'])}")
        print(f"     Concepts: {len(self.nodes['concepts'])}")
        print(f"     Authors: {len(self.nodes['authors'])}")
        print(f"     Edges: {sum(len(e) for e in self.edges.values())}")
    
    def _extract_concepts(self, title: str) -> List[str]:
        """Extract key concepts from paper title"""
        
        # Common ML/AI concepts
        concept_keywords = [
            'transformer', 'attention', 'neural', 'deep learning', 'cnn', 'rnn',
            'reinforcement', 'supervised', 'unsupervised', 'gan', 'vae',
            'bert', 'gpt', 'llm', 'diffusion', 'stable diffusion',
            'quantum', 'protein', 'crispr', 'gene', 'molecular',
            'optimization', 'gradient', 'backprop', 'training'
        ]
        
        found_concepts = []
        for concept in concept_keywords:
            if concept in title:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _build_concept_relationships(self):
        """Build edges between related concepts (co-occurrence)"""
        
        # Concepts are related if they appear in same papers
        concept_cooccurrence = defaultdict(set)
        
        for concept, paper_ids in self.nodes['concepts'].items():
            for other_concept, other_papers in self.nodes['concepts'].items():
                if concept != other_concept:
                    # Check co-occurrence
                    overlap = set(paper_ids) & set(other_papers)
                    if len(overlap) >= 2:  # At least 2 papers mention both
                        concept_cooccurrence[concept].add(other_concept)
                        self.edges['related_to'].append((concept, other_concept))
    
    def find_related_papers(self, paper_id: str, max_hops: int = 2) -> List[str]:
        """
        Find papers related through knowledge graph
        
        Args:
            paper_id: Source paper
            max_hops: Maximum graph distance
        
        Returns:
            List of related paper IDs
        """
        
        if paper_id not in self.nodes['papers']:
            return []
        
        related = set()
        
        # Papers sharing concepts
        paper_concepts = [edge[1] for edge in self.edges['builds_on'] if edge[0] == paper_id]
        for concept in paper_concepts:
            related.update(self.nodes['concepts'].get(concept, []))
        
        # Papers by same authors
        paper_authors = [edge[1] for edge in self.edges['authored_by'] if edge[0] == paper_id]
        for author in paper_authors:
            related.update(self.nodes['authors'].get(author, []))
        
        # Remove self
        related.discard(paper_id)
        
        return list(related)
    
    def get_concept_network(self, concept: str) -> Dict[str, Any]:
        """Get all papers and related concepts for a concept"""
        
        if concept not in self.nodes['concepts']:
            return {'concept': concept, 'papers': [], 'related_concepts': []}
        
        papers = self.nodes['concepts'][concept]
        related = [edge[1] for edge in self.edges['related_to'] if edge[0] == concept]
        
        return {
            'concept': concept,
            'papers': papers,
            'paper_count': len(papers),
            'related_concepts': list(set(related))
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        
        return {
            'total_papers': len(self.nodes['papers']),
            'total_concepts': len(self.nodes['concepts']),
            'total_authors': len(self.nodes['authors']),
            'total_edges': sum(len(e) for e in self.edges.values()),
            'avg_concepts_per_paper': sum(len(papers) for papers in self.nodes['concepts'].values()) / max(len(self.nodes['papers']), 1),
            'most_common_concepts': sorted(
                self.nodes['concepts'].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
        }