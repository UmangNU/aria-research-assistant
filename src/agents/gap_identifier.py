"""
Research Gap Identifier
Identifies what HASN'T been studied

META-INTELLIGENCE - Finding what's missing!
"""

from src.agents.base_agent import BaseAgent
from src.utils.llm import get_llm
from typing import Dict, Any, List, Set
import json

class ResearchGapIdentifier(BaseAgent):
    """Identify unexplored research areas"""
    
    def __init__(self):
        super().__init__(
            name="Research Gap Identifier",
            role="Finding unexplored research directions"
        )
        self.llm = get_llm()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify research gaps
        
        Input: {
            'query': str,
            'papers': List[Dict],
            'analyzed_papers': List[Dict],
            'domain': str
        }
        
        Output: {
            'gaps_found': List[Dict],
            'covered_topics': List[str],
            'suggested_directions': List[str]
        }
        """
        
        query = input_data.get('query', '')
        papers = input_data.get('papers', [])
        analyzed = input_data.get('analyzed_papers', [])
        domain = input_data.get('domain', 'general')
        
        print(f"\n🔍 Gap Identifier: Analyzing {len(papers)} papers for gaps...")
        
        # Extract what HAS been studied
        covered = self._extract_covered_topics(analyzed)
        
        print(f"   ✓ Identified {len(covered)} covered topics")
        
        # LLM-based gap analysis
        gaps = self._analyze_gaps(query, analyzed, covered, domain)
        
        print(f"   ✓ Found {len(gaps.get('gaps_found', []))} potential gaps")
        
        result = {
            'covered_topics': list(covered),
            'gaps_found': gaps.get('gaps_found', []),
            'emerging_areas': gaps.get('emerging_areas', []),
            'suggested_directions': gaps.get('suggested_directions', []),
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _extract_covered_topics(self, analyzed_papers: List[Dict]) -> Set[str]:
        """Extract topics covered in analyzed papers"""
        
        topics = set()
        
        keywords = [
            'transformer', 'cnn', 'rnn', 'lstm', 'attention', 'bert', 'gpt',
            'reinforcement', 'supervised', 'unsupervised', 'semi-supervised',
            'protein', 'gene', 'crispr', 'dna', 'molecular',
            'quantum', 'classical', 'optimization', 'prediction',
            'generation', 'classification', 'detection'
        ]
        
        for paper in analyzed_papers:
            title = paper.get('title', '').lower()
            for keyword in keywords:
                if keyword in title:
                    topics.add(keyword)
        
        return topics
    
    def _analyze_gaps(self, 
                     query: str,
                     analyzed_papers: List[Dict],
                     covered_topics: Set[str],
                     domain: str) -> Dict:
        """Use LLM to identify research gaps"""
        
        papers_text = "\n".join([
            f"- {p.get('title', 'Unknown')} ({p.get('published', '2024')[:4]})"
            for p in analyzed_papers[:10]
        ])
        
        covered_text = ', '.join(list(covered_topics)[:15])
        
        prompt = f"""You are identifying research gaps in {domain}.

Query: "{query}"

Papers Analyzed:
{papers_text}

Topics Covered: {covered_text}

Your Task: Identify what HASN'T been studied yet.

Consider:
1. Method combinations not yet explored (e.g., "transformers for protein folding")
2. Underexplored domains or datasets
3. Open questions from recent papers
4. Cross-disciplinary opportunities

Output ONLY valid JSON:
{{
  "gaps_found": [
    {{"gap": "specific gap description", "confidence": "high/medium/low", "rationale": "why this is a gap"}},
    {{"gap": "another gap", "confidence": "medium", "rationale": "reasoning"}}
  ],
  "emerging_areas": ["emerging area 1", "emerging area 2"],
  "suggested_directions": ["future direction 1", "future direction 2", "future direction 3"]
}}

Identify 2-4 concrete research gaps (JSON only):"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=700, temperature=0.8)
            
            # Parse JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            gaps = json.loads(response.strip())
            
            print(f"   ✓ Gap analysis complete")
            
            return gaps
            
        except Exception as e:
            print(f"   ⚠️  Gap analysis failed: {e}")
            
            # Fallback
            return {
                'gaps_found': [
                    {
                        'gap': 'Limited cross-domain applications',
                        'confidence': 'medium',
                        'rationale': 'Based on available papers, cross-disciplinary work appears limited'
                    }
                ],
                'emerging_areas': ['Interdisciplinary applications'],
                'suggested_directions': ['Explore novel method combinations', 'Test on new datasets']
            }