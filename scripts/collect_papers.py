# scripts/collect_papers.py
"""
Data Collection Script for ARIA Research Assistant
Collects papers from arXiv API for initial RAG database

Usage:
    python scripts/collect_papers.py
    
Output:
    data/papers/arxiv_papers.json
"""

import arxiv
import json
import os
from datetime import datetime
from typing import List, Dict
import time

def collect_arxiv_papers(domains: Dict[str, str], papers_per_domain: int = 200) -> List[Dict]:
    """
    Collect papers from arXiv for initial RAG database
    
    Args:
        domains: Dictionary mapping domain names to arXiv query strings
        papers_per_domain: Number of papers to collect per domain
        
    Returns:
        List of paper dictionaries
    """
    
    all_papers = []
    
    for domain_name, query in domains.items():
        print(f"\n{'='*60}")
        print(f"Collecting papers for domain: {domain_name}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=papers_per_domain,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            domain_papers = []
            for i, paper in enumerate(search.results()):
                paper_data = {
                    'id': paper.entry_id,
                    'title': paper.title,
                    'abstract': paper.summary.replace('\n', ' '),  # Clean newlines
                    'authors': [author.name for author in paper.authors],
                    'published': paper.published.isoformat(),
                    'updated': paper.updated.isoformat(),
                    'categories': paper.categories,
                    'domain': domain_name,
                    'pdf_url': paper.pdf_url,
                    'primary_category': paper.primary_category,
                }
                domain_papers.append(paper_data)
                all_papers.append(paper_data)
                
                # Progress update every 50 papers
                if (i + 1) % 50 == 0:
                    print(f"  Collected {i + 1}/{papers_per_domain} papers for {domain_name}...")
                
                # Be nice to arXiv API
                time.sleep(0.1)
            
            print(f"✓ Completed {domain_name}: {len(domain_papers)} papers")
            print(f"  Total papers so far: {len(all_papers)}")
            
        except Exception as e:
            print(f"✗ Error collecting papers for {domain_name}: {e}")
            continue
    
    return all_papers


def save_papers(papers: List[Dict], output_file: str = 'data/papers/arxiv_papers.json'):
    """Save papers to JSON file"""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Papers saved to: {output_file}")
    print(f"✓ Total papers: {len(papers)}")
    print(f"{'='*60}")


def print_statistics(papers: List[Dict]):
    """Print collection statistics"""
    
    from collections import Counter
    
    print(f"\n{'='*60}")
    print("COLLECTION STATISTICS")
    print(f"{'='*60}")
    
    # Papers per domain
    domain_counts = Counter(p['domain'] for p in papers)
    print("\nPapers per domain:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count}")
    
    # Total authors
    all_authors = [author for p in papers for author in p['authors']]
    unique_authors = set(all_authors)
    print(f"\nTotal authors: {len(all_authors)}")
    print(f"Unique authors: {len(unique_authors)}")
    
    # Date range
    dates = [p['published'] for p in papers]
    print(f"\nDate range:")
    print(f"  Earliest: {min(dates)[:10]}")
    print(f"  Latest: {max(dates)[:10]}")
    
    # Categories
    all_categories = [cat for p in papers for cat in p['categories']]
    top_categories = Counter(all_categories).most_common(10)
    print(f"\nTop 10 categories:")
    for cat, count in top_categories:
        print(f"  {cat}: {count}")


def main():
    """Main execution function"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         ARIA Research Assistant - Data Collection         ║
    ║                   arXiv Paper Scraper                     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Define domains and queries
    domains = {
        'cs_ml': 'cat:cs.LG OR cat:cs.AI',
        'cs_nlp': 'cat:cs.CL',
        'cs_cv': 'cat:cs.CV',
        'biology': 'cat:q-bio.*',
        'physics': 'cat:physics.comp-ph OR cat:cond-mat.stat-mech',
        'medicine': 'all:medical OR all:clinical OR all:disease'
    }
    
    papers_per_domain = 200  # Adjust this if you want more/fewer papers
    
    print(f"Configuration:")
    print(f"  Domains: {len(domains)}")
    print(f"  Papers per domain: {papers_per_domain}")
    print(f"  Expected total: ~{len(domains) * papers_per_domain} papers")
    print(f"\nStarting collection...\n")
    
    start_time = time.time()
    
    # Collect papers
    papers = collect_arxiv_papers(domains, papers_per_domain)
    
    # Save papers
    save_papers(papers)
    
    # Print statistics
    print_statistics(papers)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Collection completed in {elapsed:.1f} seconds")
    print(f"✓ Average: {elapsed/len(papers):.2f} seconds per paper")


if __name__ == "__main__":
    main()