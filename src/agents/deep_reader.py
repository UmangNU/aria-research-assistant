# src/agents/deep_reader.py
from src.agents.base_agent import BaseAgent
from src.utils.llm import get_llm
from src.prompts.prompt_library import PromptLibrary
from typing import Dict, Any, List

class DeepReaderAgent(BaseAgent):
    """Analyzes papers using advanced prompt engineering"""
    
    def __init__(self):
        super().__init__(
            name="Deep Reader",
            role="Analyzing papers with chain-of-thought reasoning"
        )
        self.llm = get_llm()
        self.prompt_lib = PromptLibrary()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze papers using advanced prompts
        
        Input: {
            'papers': List[Dict],
            'depth': str,
            'query': str,
            'domain': str (optional),
            'query_type': str (optional)
        }
        """
        papers = input_data.get('papers', [])
        depth = input_data.get('depth', 'moderate')
        query = input_data.get('query', '')
        domain = input_data.get('domain', 'general')
        query_type = input_data.get('query_type', 'exploratory')
        
        # Determine papers to analyze
        if depth == 'shallow':
            num_papers = min(3, len(papers))
        elif depth == 'moderate':
            num_papers = min(5, len(papers))
        else:  # deep
            num_papers = min(8, len(papers))
        
        papers_to_analyze = papers[:num_papers]
        
        # Build advanced prompt
        print(f"   📖 Building advanced prompt (CoT + Few-shot)...")
        advanced_prompt = self.prompt_lib.build_analysis_prompt(
            papers=papers_to_analyze,
            query=query,
            domain=domain,
            query_type=query_type,
            depth=depth
        )
        
        # Optimize for token budget
        advanced_prompt = self.prompt_lib.optimize_for_tokens(advanced_prompt, max_tokens=3000)
        
        # Generate analysis with advanced prompt
        print(f"   🤖 LLM analyzing with chain-of-thought...")
        analysis_text = self.llm.generate(advanced_prompt, max_tokens=1000, temperature=0.7)
        
        # Structure results
        analyzed_papers = []
        for paper in papers_to_analyze:
            analyzed_papers.append({
                'title': paper['metadata']['title'],
                'relevance_score': paper.get('score', 0),
                'credibility_score': paper.get('credibility', {}).get('credibility_score', 0),
                'domain': paper['metadata']['domain'],
                'published': paper['metadata']['published']
            })
        
        result = {
            'analyzed_papers': analyzed_papers,
            'key_insights': analysis_text,
            'papers_analyzed': num_papers,
            'depth': depth,
            'prompt_type': 'advanced_cot_fewshot',
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result