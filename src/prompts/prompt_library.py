# src/prompts/prompt_library.py
"""
Advanced Prompt Engineering Library for ARIA
Implements systematic prompting strategies with:
- Chain-of-thought reasoning
- Few-shot learning per domain
- Self-consistency
- Context management
"""

from typing import Dict, List, Any
import json

class PromptLibrary:
    """Systematic prompt engineering for research tasks"""
    
    def __init__(self):
        self.domain_examples = self._load_domain_examples()
        self.cot_templates = self._load_cot_templates()
        
    def _load_domain_examples(self) -> Dict[str, List[Dict]]:
        """Load few-shot examples for each domain"""
        return {
            'cs_ml': [
                {
                    'query': 'What is deep learning?',
                    'analysis': 'Deep learning uses multi-layer neural networks to learn hierarchical representations. Key papers focus on: 1) Architectural innovations (CNNs, RNNs, Transformers), 2) Training techniques (backpropagation, optimization), 3) Applications (vision, NLP, RL).',
                },
                {
                    'query': 'How do transformers work?',
                    'analysis': 'Transformers use self-attention mechanisms to process sequences in parallel. Core innovations: 1) Scaled dot-product attention, 2) Multi-head attention for different representation subspaces, 3) Position encoding for sequence order.',
                }
            ],
            'biology': [
                {
                    'query': 'What is CRISPR?',
                    'analysis': 'CRISPR-Cas9 enables precise genome editing. Key aspects: 1) Guide RNA directs Cas9 to target DNA sequence, 2) Double-strand breaks enable insertion/deletion, 3) Applications in gene therapy, agriculture, disease modeling.',
                },
                {
                    'query': 'How does protein folding work?',
                    'analysis': 'Proteins fold into 3D structures determining function. Research areas: 1) Thermodynamic principles (energy minimization), 2) Prediction methods (AlphaFold, molecular dynamics), 3) Misfolding diseases (Alzheimer\'s, Parkinson\'s).',
                }
            ],
            'physics': [
                {
                    'query': 'What is quantum computing?',
                    'analysis': 'Quantum computers leverage superposition and entanglement for computation. Key topics: 1) Qubit implementations (superconducting, ion traps), 2) Quantum algorithms (Shor, Grover), 3) Error correction challenges.',
                }
            ]
        }
    
    def _load_cot_templates(self) -> Dict[str, str]:
        """Chain-of-thought templates for different query types"""
        return {
            'literature_review': """Let's approach this systematically:

1. First, I'll identify the core topic and subtopics
2. Then, I'll categorize papers by their contribution type
3. Next, I'll extract key findings from each category
4. Finally, I'll synthesize into a coherent narrative

Let me analyze:""",
            
            'methodology': """To explain this methodology, I'll break it down:

1. First, understand the problem this method solves
2. Then, examine the core technical approach
3. Next, identify key innovations and advantages
4. Finally, discuss applications and limitations

Analysis:""",
            
            'comparison': """For a thorough comparison, I'll:

1. Define evaluation criteria (accuracy, speed, applicability)
2. Analyze each approach's strengths
3. Identify weaknesses and trade-offs
4. Synthesize when to use which approach

Comparison:""",
            
            'definition': """To define this concept comprehensively:

1. Provide formal definition from foundational papers
2. Explain key components and mechanisms
3. Distinguish from related concepts
4. Give concrete examples

Definition:"""
        }
    
    def build_analysis_prompt(self, 
                             papers: List[Dict], 
                             query: str, 
                             domain: str,
                             query_type: str,
                             depth: str) -> str:
        """
        Build advanced analysis prompt with CoT and few-shot examples
        
        Args:
            papers: List of papers to analyze
            query: Research query
            domain: Domain classification
            query_type: Type of query
            depth: Analysis depth
        
        Returns:
            Optimized prompt string
        """
        
        # Select few-shot examples
        examples = self.domain_examples.get(domain, self.domain_examples['cs_ml'])[:2]
        examples_text = "\n\n".join([
            f"Example Query: {ex['query']}\nExample Analysis: {ex['analysis']}"
            for ex in examples
        ])
        
        # Select CoT template
        cot = self.cot_templates.get(query_type, self.cot_templates['literature_review'])
        
        # Build paper context
        papers_text = "\n\n".join([
            f"Paper {i+1}:\n"
            f"Title: {p['metadata']['title']}\n"
            f"Domain: {p['metadata']['domain']}\n"
            f"Year: {p['metadata']['published'][:4]}\n"
            f"Credibility: {p.get('credibility', {}).get('credibility_score', 0):.2f}"
            for i, p in enumerate(papers[:8])
        ])
        
        # Depth-specific instructions
        depth_instructions = {
            'shallow': "Provide a concise 3-4 sentence overview focusing on the most important finding.",
            'moderate': "Provide a comprehensive analysis with 4-5 key insights, identifying main themes and methodologies.",
            'deep': "Provide an in-depth analysis with 6-8 detailed insights, including methodology details, theoretical foundations, empirical results, and implications. Identify connections between papers and evaluate the strength of evidence."
        }
        
        # Construct full prompt
        prompt = f"""You are an expert research analyst with deep knowledge in {domain}.

**Few-Shot Examples:**
{examples_text}

**Task:** Analyze the following papers for the query: "{query}"

**Papers:**
{papers_text}

**Chain-of-Thought Reasoning:**
{cot}

**Depth Requirement:** {depth_instructions[depth]}

**Output Requirements:**
1. Start with the most significant finding
2. Support claims with specific paper references
3. Identify common themes across papers
4. Note any contradictions or debates
5. Mention novel methodologies
6. Assess the current state of research in this area

Provide your analysis:"""

        return prompt
    
    def build_synthesis_prompt(self,
                              query: str,
                              papers: List[Dict],
                              analysis: str,
                              style: str,
                              domain: str) -> str:
        """
        Build advanced synthesis prompt with structure guidance
        
        Args:
            query: Research query
            papers: Analyzed papers
            analysis: Analysis from Deep Reader
            style: Synthesis style
            domain: Domain classification
        
        Returns:
            Optimized synthesis prompt
        """
        
        papers_list = "\n".join([
            f"{i+1}. {p['metadata']['title']} ({p['metadata']['published'][:4]}) "
            f"[Credibility: {p.get('credibility', {}).get('credibility_score', 0):.2f}]"
            for i, p in enumerate(papers[:10])
        ])
        
        style_templates = {
            'concise': {
                'length': '2-3 paragraphs (300-400 words)',
                'structure': 'Brief overview → Key finding → Implication',
                'tone': 'Direct and clear, minimal jargon'
            },
            'detailed': {
                'length': '4-6 paragraphs (800-1000 words)',
                'structure': 'Introduction → Main findings (3-4 points) → Trends → Conclusion',
                'tone': 'Comprehensive yet accessible, balanced technical depth'
            },
            'technical': {
                'length': '6-8 paragraphs (1200-1500 words)',
                'structure': 'Context → Methodology analysis → Empirical results → Theoretical implications → Future directions',
                'tone': 'Highly technical, assume expert audience, include equations/formulas if relevant'
            }
        }
        
        template = style_templates[style]
        
        prompt = f"""You are synthesizing research findings on: "{query}"

**Domain:** {domain}

**Key Papers Analyzed:**
{papers_list}

**In-Depth Analysis:**
{analysis}

**Synthesis Requirements:**
- **Length:** {template['length']}
- **Structure:** {template['structure']}
- **Tone:** {template['tone']}

**Quality Standards:**
1. Cite specific papers when making claims
2. Identify consensus vs. ongoing debates
3. Highlight recent developments (2024-2025)
4. Note methodological trends
5. Provide actionable insights
6. Maintain academic rigor

**Self-Consistency Check:**
Before finalizing, ask yourself:
- Does this fully answer the query?
- Are all major papers incorporated?
- Is the narrative coherent?
- Would a researcher find this valuable?

Generate a high-quality research synthesis:"""

        return prompt
    
    def build_self_critique_prompt(self, summary: str, query: str) -> str:
        """
        Build prompt for self-reflection and critique
        
        Args:
            summary: Generated summary
            query: Original query
        
        Returns:
            Critique prompt
        """
        
        prompt = f"""You are a critical peer reviewer evaluating a research summary.

**Original Query:** "{query}"

**Summary to Review:**
{summary}

**Evaluation Criteria:**
1. **Completeness:** Does it fully address the query? What's missing?
2. **Accuracy:** Are claims well-supported by papers? Any overgeneralizations?
3. **Clarity:** Is it well-structured and easy to follow?
4. **Depth:** Sufficient technical detail? Too shallow or too dense?
5. **Coherence:** Logical flow? Contradictions?
6. **Novelty:** Does it identify new insights or just summarize?

**Your Task:**
Provide a critical assessment with:
- Overall quality score (0-100)
- Specific strengths (2-3 points)
- Specific weaknesses (2-3 points)
- Concrete suggestions for improvement
- Missing information that should be added

Be honest and constructive:"""

        return prompt
    
    def optimize_for_tokens(self, prompt: str, max_tokens: int = 3000) -> str:
        """
        Optimize prompt to fit within token budget
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum tokens allowed
        
        Returns:
            Optimized prompt
        """
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(prompt) <= max_chars:
            return prompt
        
        # Truncate papers section if needed
        # (More sophisticated implementation would intelligently prune)
        return prompt[:max_chars] + "\n\n[Truncated for length]"