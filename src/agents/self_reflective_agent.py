# src/agents/self_reflective_agent.py
"""
Self-Reflective Quality Improvement Agent
AI that critiques its own outputs and iteratively improves them

Meta-cognitive AI - thinks about its own thinking!
"""

from typing import Dict, Any, Tuple, List
from src.agents.base_agent import BaseAgent
from src.utils.llm import get_llm
from src.prompts.prompt_library import PromptLibrary
import json

class SelfReflectiveAgent(BaseAgent):
    """
    Agent that critiques and improves its own outputs
    
    Process:
    1. Generate initial research summary
    2. Critique: Identify weaknesses, gaps, errors
    3. Improve: Generate enhanced version addressing critique
    4. Repeat until quality threshold met or max iterations
    """
    
    def __init__(self):
        super().__init__(
            name="Self-Reflective Agent",
            role="Meta-cognitive quality improvement through self-critique"
        )
        self.llm = get_llm()
        self.prompt_lib = PromptLibrary()
        self.quality_threshold = 85  # Out of 100
        self.max_iterations = 2
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute self-reflective improvement
        
        Input: {
            'initial_summary': str,
            'query': str,
            'papers': List (titles or dicts),
            'max_iterations': int (optional)
        }
        
        Output: {
            'initial_summary': str,
            'critiques': List[Dict],
            'improved_summary': str,
            'improvement_iterations': int,
            'quality_progression': List[int],
            'agent': str
        }
        """
        initial_summary = input_data.get('initial_summary', '')
        query = input_data.get('query', '')
        papers = input_data.get('papers', [])
        max_iters = input_data.get('max_iterations', self.max_iterations)
        
        print(f"\n🪞 Self-Reflection: Starting quality improvement cycle...")
        
        current_summary = initial_summary
        critiques = []
        quality_scores = []
        iteration = 0
        
        # Iterative improvement loop
        while iteration < max_iters:
            iteration += 1
            
            print(f"\n   Iteration {iteration}/{max_iters}:")
            
            # Step 1: Critique current summary
            print(f"     🔍 Critiquing current version...")
            critique = self._critique_summary(current_summary, query)
            critiques.append(critique)
            
            quality_score = critique['quality_score']
            quality_scores.append(quality_score)
            
            print(f"     ✓ Quality Score: {quality_score}/100")
            print(f"     ✓ Identified {len(critique.get('weaknesses', []))} weaknesses")
            
            # Check if threshold met
            if quality_score >= self.quality_threshold:
                print(f"     ✅ Quality threshold met ({quality_score} >= {self.quality_threshold})")
                break
            
            # Step 2: Improve based on critique
            print(f"     🔧 Generating improved version...")
            current_summary = self._improve_summary(
                current_summary, 
                query, 
                critique,
                papers
            )
            
            print(f"     ✓ Improved version generated ({len(current_summary)} chars)")
        
        # Final comparison
        if len(quality_scores) > 1:
            improvement_pct = ((quality_scores[-1] - quality_scores[0]) / quality_scores[0] * 100)
        else:
            improvement_pct = 0
        
        print(f"\n   📈 Quality improvement: {quality_scores[0]} → {quality_scores[-1]} (+{improvement_pct:.1f}%)")
        
        result = {
            'initial_summary': initial_summary,
            'critiques': critiques,
            'improved_summary': current_summary,
            'improvement_iterations': iteration,
            'quality_progression': quality_scores,
            'initial_quality': quality_scores[0] if quality_scores else 0,
            'final_quality': quality_scores[-1] if quality_scores else 0,
            'improvement_percent': round(improvement_pct, 2),
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _critique_summary(self, summary: str, query: str) -> Dict[str, Any]:
        """
        Generate critical assessment of summary
        
        Args:
            summary: Summary to critique
            query: Original query
        
        Returns:
            Critique with scores and specific feedback
        """
        
        critique_prompt = self.prompt_lib.build_self_critique_prompt(summary, query)
        
        critique_prompt += """

Output as JSON:
{
  "quality_score": 75,
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "missing_information": ["what's missing"]
}

Provide your critical assessment:"""

        try:
            response = self.llm.generate(critique_prompt, max_tokens=600, temperature=0.3)
            
            # Parse JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            critique = json.loads(response.strip())
            
            # Ensure required fields
            critique.setdefault('quality_score', 70)
            critique.setdefault('strengths', [])
            critique.setdefault('weaknesses', [])
            critique.setdefault('suggestions', [])
            critique.setdefault('missing_information', [])
            
            return critique
            
        except Exception as e:
            print(f"       ⚠️  Critique parsing failed: {e}")
            return {
                'quality_score': 75,
                'strengths': ['Generally well-structured'],
                'weaknesses': ['Could be more specific'],
                'suggestions': ['Add more details'],
                'missing_information': []
            }
    
    def _improve_summary(self,
                        current_summary: str,
                        query: str,
                        critique: Dict[str, Any],
                        papers: Any) -> str:
        """
        Generate improved summary addressing critique
        
        Args:
            current_summary: Current version
            query: Original query
            critique: Critique results
            papers: Available papers (strings or dicts)
        
        Returns:
            Improved summary
        """
        
        # Handle both string titles and dict papers
        if not papers:
            papers_context = "No additional papers available"
        elif isinstance(papers, list):
            if len(papers) > 0 and isinstance(papers[0], str):
                # Papers are titles (strings)
                papers_context = "\n".join([f"- {p}" for p in papers[:5]])
            else:
                # Papers are dicts
                papers_context = "\n".join([
                    f"- {p.get('title', 'Unknown')}" if isinstance(p, dict) else f"- {p}"
                    for p in papers[:5]
                ])
        else:
            papers_context = str(papers)
        
        weaknesses_text = ', '.join(critique.get('weaknesses', ['None']))
        suggestions_text = ', '.join(critique.get('suggestions', ['None']))
        missing_text = ', '.join(critique.get('missing_information', ['None']))
        
        prompt = f"""You are improving a research summary based on critical feedback.

Original Query: "{query}"

Current Summary:
{current_summary}

Critical Assessment:
- Quality Score: {critique.get('quality_score', 70)}/100
- Weaknesses: {weaknesses_text}
- Suggestions: {suggestions_text}
- Missing Information: {missing_text}

Available Papers for Context:
{papers_context}

Your task: Generate an improved version that:
1. Addresses ALL identified weaknesses
2. Implements ALL suggestions
3. Adds any missing information
4. Maintains existing strengths
5. Improves clarity, completeness, and academic rigor

Write the improved summary (aim for quality score 90+):"""

        improved = self.llm.generate(prompt, max_tokens=1500, temperature=0.7)
        
        return improved