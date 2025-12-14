# src/agents/synthesizer.py
from src.agents.base_agent import BaseAgent
from src.utils.llm import get_llm
from src.prompts.prompt_library import PromptLibrary
from typing import Dict, Any, List

class SynthesizerAgent(BaseAgent):
    """Synthesizes research using advanced prompt engineering"""
    
    def __init__(self):
        super().__init__(
            name="Synthesizer",
            role="Creating research summaries with structured prompting"
        )
        self.llm = get_llm()
        self.prompt_lib = PromptLibrary()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize with advanced prompts
        
        Input: {
            'query': str,
            'analyzed_papers': List[Dict],
            'key_insights': str,
            'style': str,
            'domain': str (optional)
        }
        """
        query = input_data.get('query', '')
        analyzed_papers = input_data.get('analyzed_papers', [])
        key_insights = input_data.get('key_insights', '')
        style = input_data.get('style', 'detailed')
        domain = input_data.get('domain', 'general')
        
        # Convert to format prompt expects
        papers_for_prompt = [
            {'metadata': p} for p in analyzed_papers
        ]
        
        # Build advanced synthesis prompt
        print(f"   ✍️  Building advanced synthesis prompt...")
        advanced_prompt = self.prompt_lib.build_synthesis_prompt(
            query=query,
            papers=papers_for_prompt,
            analysis=key_insights,
            style=style,
            domain=domain
        )
        
        # Optimize for tokens
        advanced_prompt = self.prompt_lib.optimize_for_tokens(advanced_prompt, max_tokens=4000)
        
        # Generate summary
        print(f"   🤖 LLM synthesizing with structured prompting...")
        summary = self.llm.generate(advanced_prompt, max_tokens=1500, temperature=0.7)
        
        # NEW: Format summary for professional display
        summary = self._format_summary_for_display(summary)
        
        # Extract key papers
        key_papers = [p['title'] for p in analyzed_papers[:5]]
        
        # Generate citations
        citations = [
            f"{p['title']} ({p['published'][:4]})"
            for p in analyzed_papers[:10]
        ]
        
        result = {
            'summary': summary,
            'key_papers': key_papers,
            'citations': citations,
            'style': style,
            'prompt_type': 'advanced_structured',
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _format_summary_for_display(self, summary: str) -> str:
        """
        Format summary with professional structure and emoji headers
        
        Args:
            summary: Raw LLM output
        
        Returns:
            Formatted markdown summary with sections
        """
        
        formatted = summary
        
        # Replace section headers with emoji-enhanced versions
        replacements = {
            # Introduction variations
            'Introduction': '## 📌 Introduction',
            'Context': '## 📌 Context',
            'Background': '## 📌 Background',
            
            # Findings variations
            'Key Findings': '## 🔍 Key Findings',
            'Main Findings': '## 🔍 Main Findings',
            'Findings': '## 🔍 Findings',
            
            # Methodology variations
            'Methodology Analysis': '## 🔬 Methodology Analysis',
            'Methodology': '## 🔬 Methodology',
            'Methods': '## 🔬 Methods',
            'Methodological': '## 🔬 Methodological Analysis',
            
            # Results variations
            'Empirical Results': '## 📊 Empirical Results',
            'Results': '## 📊 Results',
            'Evidence': '## 📊 Evidence',
            
            # Trends variations
            'Current Trends': '## 📈 Current Trends',
            'Trends': '## 📈 Trends',
            'Recent Developments': '## 📈 Recent Developments',
            
            # Future variations
            'Future Directions': '## 🚀 Future Directions',
            'Future Work': '## 🚀 Future Work',
            'Future Research': '## 🚀 Future Research',
            
            # Conclusion variations
            'Conclusion': '## ✅ Conclusion',
            'Summary': '## ✅ Summary',
            'In summary': '## ✅ Summary',
            'In conclusion': '## ✅ Conclusion'
        }
        
        # Apply replacements (case-sensitive first)
        for old, new in replacements.items():
            if old in formatted:
                formatted = formatted.replace(old, new)
        
        # Add spacing around headers
        formatted = formatted.replace('##', '\n\n##')
        
        # Clean up excessive newlines
        while '\n\n\n' in formatted:
            formatted = formatted.replace('\n\n\n', '\n\n')
        
        return formatted.strip()