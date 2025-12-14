"""
Comparative Analysis Agent
Side-by-side comparison of research approaches

ADVANCED SYNTHESIS - Goes beyond retrieval!
"""

from src.agents.base_agent import BaseAgent
from src.utils.llm import get_llm
from typing import Dict, Any, List
import json

class ComparativeAnalyzer(BaseAgent):
    """Compare multiple research approaches side-by-side"""
    
    def __init__(self):
        super().__init__(
            name="Comparative Analyzer",
            role="Side-by-side comparison of research approaches"
        )
        self.llm = get_llm()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparative analysis
        
        Input: {
            'query': str (must contain comparison keywords),
            'papers': List[Dict],
            'analyzed_papers': List[Dict] (from Deep Reader)
        }
        
        Output: {
            'is_comparison': bool,
            'entity1': str,
            'entity2': str,
            'comparison_table': Dict,
            'narrative': str,
            'when_to_use': Dict,
            'winner': str
        }
        """
        
        query = input_data.get('query', '')
        papers = input_data.get('papers', [])
        analyzed = input_data.get('analyzed_papers', [])
        
        # Check if this is a comparison query
        entities = self._extract_comparison_entities(query)
        
        if len(entities) < 2:
            return {
                'is_comparison': False,
                'message': 'Not a comparison query',
                'agent': self.name
            }
        
        entity1, entity2 = entities[0], entities[1]
        
        print(f"\n🔄 Comparative Analysis: {entity1} vs {entity2}")
        
        # Build comparison
        comparison = self._generate_comparison(query, entity1, entity2, papers, analyzed)
        
        comparison['is_comparison'] = True
        comparison['entity1'] = entity1
        comparison['entity2'] = entity2
        comparison['agent'] = self.name
        
        self.add_to_memory(comparison)
        return comparison
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extract what's being compared"""
        
        query_lower = query.lower()
        
        # Pattern 1: "X vs Y" or "X versus Y"
        if ' vs ' in query_lower or ' versus ' in query_lower:
            parts = query_lower.replace(' versus ', ' vs ').split(' vs ')
            if len(parts) >= 2:
                # Extract key terms before and after 'vs'
                before = parts[0].strip().split()
                after = parts[1].strip().split()
                
                # Take last 1-2 words before, first 1-2 words after
                entity1 = ' '.join(before[-2:] if len(before) >= 2 else before)
                entity2 = ' '.join(after[:2])
                
                return [entity1.strip(), entity2.strip()]
        
        # Pattern 2: "compare X and Y"
        if 'compare' in query_lower and ' and ' in query_lower:
            after_compare = query_lower.split('compare')[-1]
            if ' and ' in after_compare:
                parts = after_compare.split(' and ')
                entity1 = parts[0].strip().split()[:2]
                entity2 = parts[1].strip().split()[:2]
                return [' '.join(entity1), ' '.join(entity2)]
        
        # Pattern 3: "difference between X and Y"
        if 'difference' in query_lower and ' and ' in query_lower:
            after_diff = query_lower.split('between')[-1] if 'between' in query_lower else query_lower
            if ' and ' in after_diff:
                parts = after_diff.split(' and ')
                entity1 = ' '.join(parts[0].strip().split()[:2])
                entity2 = ' '.join(parts[1].strip().split()[:2])
                return [entity1, entity2]
        
        return []
    
    def _generate_comparison(self, 
                            query: str,
                            entity1: str, 
                            entity2: str,
                            papers: List[Dict],
                            analyzed: List[Dict]) -> Dict:
        """Generate detailed comparison using LLM"""
        
        papers_context = "\n".join([
            f"{i+1}. {p.get('title', 'Unknown')} ({p.get('published', '2024')[:4]})"
            for i, p in enumerate(analyzed[:8])
        ])
        
        prompt = f"""You are conducting a comprehensive comparative analysis.

Query: "{query}"

Comparing: {entity1.upper()} vs {entity2.upper()}

Research Papers:
{papers_context}

Your Task: Create a detailed comparison covering:
1. Performance/Effectiveness
2. Computational Cost/Efficiency  
3. Data Requirements
4. Interpretability
5. Use Cases/Applications
6. Strengths & Weaknesses

Output ONLY valid JSON:
{{
  "comparison_table": {{
    "Performance": {{"{entity1}": "description", "{entity2}": "description"}},
    "Cost": {{"{entity1}": "description", "{entity2}": "description"}},
    "Data Requirements": {{"{entity1}": "description", "{entity2}": "description"}},
    "Interpretability": {{"{entity1}": "description", "{entity2}": "description"}},
    "Use Cases": {{"{entity1}": "description", "{entity2}": "description"}}
  }},
  "narrative": "2-3 paragraph comparison highlighting key differences and trade-offs",
  "when_to_use": {{
    "{entity1}": ["use case 1", "use case 2", "use case 3"],
    "{entity2}": ["use case 1", "use case 2", "use case 3"]
  }},
  "winner": "entity1/entity2/tie/depends",
  "rationale": "brief explanation of winner choice"
}}

Generate comparison (JSON only):"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=1200, temperature=0.7)
            
            # Parse JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            comparison = json.loads(response.strip())
            
            print(f"   ✓ Comparison generated successfully")
            
            return comparison
            
        except Exception as e:
            print(f"   ⚠️  Comparison generation failed: {e}")
            
            # Fallback comparison
            return {
                'comparison_table': {
                    'Overview': {
                        entity1: f"Research shows various approaches for {entity1}",
                        entity2: f"Literature discusses applications of {entity2}"
                    }
                },
                'narrative': f"Based on the analyzed papers, both {entity1} and {entity2} have distinct characteristics and use cases in their respective domains.",
                'when_to_use': {
                    entity1: ["Context-specific applications"],
                    entity2: ["Alternative approaches"]
                },
                'winner': 'depends',
                'rationale': 'Choice depends on specific requirements and constraints'
            }
    
    def format_as_markdown(self, comparison: Dict) -> str:
        """Format comparison as markdown table"""
        
        if not comparison.get('is_comparison', False):
            return ""
        
        entity1 = comparison['entity1']
        entity2 = comparison['entity2']
        table = comparison.get('comparison_table', {})
        
        md = f"## {entity1.title()} vs {entity2.title()}\n\n"
        
        if table:
            md += "| Aspect | " + entity1.title() + " | " + entity2.title() + " |\n"
            md += "|--------|" + "-" * (len(entity1) + 2) + "|" + "-" * (len(entity2) + 2) + "|\n"
            
            for aspect, values in table.items():
                val1 = values.get(entity1, values.get(entity1.title(), 'N/A'))
                val2 = values.get(entity2, values.get(entity2.title(), 'N/A'))
                md += f"| **{aspect}** | {val1} | {val2} |\n"
        
        md += f"\n### Analysis\n\n{comparison.get('narrative', '')}\n"
        
        when_to_use = comparison.get('when_to_use', {})
        if when_to_use:
            md += f"\n### When to Use {entity1.title()}\n"
            for use_case in when_to_use.get(entity1, when_to_use.get(entity1.title(), [])):
                md += f"- {use_case}\n"
            
            md += f"\n### When to Use {entity2.title()}\n"
            for use_case in when_to_use.get(entity2, when_to_use.get(entity2.title(), [])):
                md += f"- {use_case}\n"
        
        return md