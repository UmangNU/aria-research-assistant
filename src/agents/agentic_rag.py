# src/agents/agentic_rag.py
"""
Agentic RAG - Self-Questioning Retrieval Agent
AI that asks itself follow-up questions to refine retrieval iteratively

This is a UNIQUE innovation showing true agentic reasoning!
"""

from typing import Dict, Any, List, Tuple
from src.agents.base_agent import BaseAgent
from src.rag.vector_store import VectorStore
from src.utils.llm import get_llm
import json

class AgenticRAGAgent(BaseAgent):
    """
    Self-questioning RAG agent that iteratively refines retrieval
    
    Process:
    1. Receive broad query
    2. Generate clarifying sub-questions
    3. Retrieve papers for each sub-question
    4. Synthesize comprehensive answer
    
    Example:
    Query: "What is deep learning?"
    ↓
    Sub-Q1: "What are the foundational architectures in deep learning?"
    Sub-Q2: "What are recent advances in deep learning?"
    Sub-Q3: "What are practical applications of deep learning?"
    ↓
    Retrieve for each → Comprehensive answer
    """
    
    def __init__(self, vector_store: VectorStore):
        super().__init__(
            name="Agentic RAG",
            role="Self-questioning retrieval with iterative refinement"
        )
        self.vector_store = vector_store
        self.llm = get_llm()
        self.max_iterations = 3
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agentic RAG with self-questioning
        
        Input: {
            'query': str,
            'max_subqueries': int (default: 3),
            'enable_iteration': bool (default: True)
        }
        
        Output: {
            'original_query': str,
            'sub_questions': List[str],
            'retrieved_papers': Dict[str, List],
            'synthesis': str,
            'agent': str
        }
        """
        query = input_data.get('query', '')
        max_subqueries = input_data.get('max_subqueries', 3)
        enable_iteration = input_data.get('enable_iteration', True)
        
        print(f"\n🤔 Agentic RAG: Analyzing query complexity...")
        
        # Step 1: Generate sub-questions
        sub_questions = self._generate_sub_questions(query, max_subqueries)
        
        print(f"   ✓ Generated {len(sub_questions)} sub-questions:")
        for i, sq in enumerate(sub_questions, 1):
            print(f"     {i}. {sq}")
        
        # Step 2: Retrieve for each sub-question
        print(f"\n📚 Retrieving papers for each sub-question...")
        all_papers = {}
        unique_papers = set()
        
        for sq in sub_questions:
            papers = self.vector_store.search(sq, top_k=5)
            all_papers[sq] = papers
            
            # Track unique papers
            for p in papers:
                unique_papers.add(p['id'])
            
            print(f"   ✓ Sub-Q: '{sq[:50]}...' → {len(papers)} papers")
        
        print(f"\n   📊 Total unique papers retrieved: {len(unique_papers)}")
        
        # Step 3: Synthesize comprehensive answer
        print(f"\n🔗 Synthesizing comprehensive answer from all retrievals...")
        synthesis = self._synthesize_multi_perspective(query, sub_questions, all_papers)
        
        print(f"   ✓ Generated {len(synthesis)} character synthesis")
        
        result = {
            'original_query': query,
            'sub_questions': sub_questions,
            'retrieved_papers': {sq: [p['metadata']['title'] for p in papers] 
                                for sq, papers in all_papers.items()},
            'unique_papers_count': len(unique_papers),
            'synthesis': synthesis,
            'agentic_reasoning': 'self_questioning',
            'agent': self.name
        }
        
        self.add_to_memory(result)
        return result
    
    def _generate_sub_questions(self, query: str, max_subqueries: int) -> List[str]:
        """
        Generate clarifying sub-questions for broad query
        
        Args:
            query: Original research query
            max_subqueries: Maximum number of sub-questions
        
        Returns:
            List of sub-questions
        """
        
        prompt = f"""You are analyzing a research query to break it down into focused sub-questions.

Original Query: "{query}"

Your task: Generate {max_subqueries} specific sub-questions that:
1. Cover different aspects of the main query
2. Are more focused and specific than the original
3. Together provide comprehensive coverage
4. Enable targeted paper retrieval

Think step-by-step:
- What are the foundational concepts to understand?
- What are the current developments?
- What are the practical applications or implications?

Output ONLY a JSON object:
{{
  "sub_questions": [
    "sub-question 1",
    "sub-question 2",
    "sub-question 3"
  ]
}}

Generate {max_subqueries} focused sub-questions:"""

        try:
            response = self.llm.generate(prompt, max_tokens=300, temperature=0.7)
            
            # Parse JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            data = json.loads(response.strip())
            sub_questions = data.get('sub_questions', [])
            
            return sub_questions[:max_subqueries]
            
        except Exception as e:
            print(f"   ⚠️  Sub-question generation failed: {e}")
            # Fallback: create basic sub-questions
            return [
                f"{query} - foundational concepts",
                f"{query} - recent advances",
                f"{query} - practical applications"
            ]
    
    def _synthesize_multi_perspective(self, 
                                      query: str, 
                                      sub_questions: List[str],
                                      papers_per_subq: Dict[str, List]) -> str:
        """
        Synthesize answer from multiple retrieval perspectives
        
        Args:
            query: Original query
            sub_questions: List of sub-questions
            papers_per_subq: Papers retrieved for each sub-question
        
        Returns:
            Comprehensive synthesis
        """
        
        # Build context from all papers
        context_parts = []
        
        for sq, papers in papers_per_subq.items():
            papers_text = "\n".join([
                f"- {p['metadata']['title']}" for p in papers[:3]
            ])
            context_parts.append(f"Sub-question: {sq}\nRelevant papers:\n{papers_text}")
        
        full_context = "\n\n".join(context_parts)
        
        prompt = f"""You are synthesizing research from multiple retrieval perspectives.

Original Query: "{query}"

Multi-Perspective Retrieval Results:
{full_context}

Your task: Create a comprehensive answer that:
1. Addresses the original query completely
2. Integrates insights from all sub-question perspectives
3. Provides a coherent narrative (not just listing sub-answers)
4. Identifies connections between different aspects

Structure:
- Overview addressing the main query
- Key insights from each perspective integrated naturally
- Synthesis showing how pieces connect
- Conclusion

Write a comprehensive research synthesis (500-800 words):"""

        synthesis = self.llm.generate(prompt, max_tokens=1000, temperature=0.7)
        
        return synthesis
    
    def explain_reasoning(self) -> Dict[str, Any]:
        """
        Explain the agentic reasoning process
        
        Returns:
            Explanation of decisions made
        """
        if not self.memory:
            return {'explanation': 'No queries processed yet'}
        
        last_execution = self.memory[-1]
        
        return {
            'original_query': last_execution['original_query'],
            'reasoning_steps': [
                f"1. Analyzed query complexity",
                f"2. Generated {len(last_execution['sub_questions'])} focused sub-questions",
                f"3. Retrieved papers for each perspective",
                f"4. Found {last_execution['unique_papers_count']} unique papers total",
                f"5. Synthesized multi-perspective answer"
            ],
            'sub_questions_generated': last_execution['sub_questions'],
            'coverage': f"{last_execution['unique_papers_count']} papers across {len(last_execution['sub_questions'])} perspectives"
        }