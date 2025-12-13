# app.py
"""
ARIA Streamlit Web Demo
Interactive research assistant with all 7 GenAI features
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.rag.vector_store import VectorStore
from src.agents.orchestrator import AgentOrchestrator
from src.agents.agentic_rag import AgenticRAGAgent
from src.agents.self_reflective_agent import SelfReflectiveAgent
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ARIA - AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    with st.spinner("Loading ARIA system..."):
        st.session_state.vector_store = VectorStore()
        with open('data/papers/arxiv_papers.json', 'r') as f:
            papers = json.load(f)
        st.session_state.vector_store.add_papers(papers[:200])
        st.session_state.orchestrator = AgentOrchestrator(st.session_state.vector_store)
        st.session_state.agentic_rag = AgenticRAGAgent(st.session_state.vector_store)
        st.session_state.reflective = SelfReflectiveAgent()

# Header
st.title("🔬 ARIA - Adaptive Research Intelligence Agent")
st.markdown("**AI research assistant with reinforcement learning + advanced GenAI features**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    top_k = st.slider("Papers to retrieve", 3, 15, 5)
    depth = st.selectbox("Analysis depth", ["shallow", "moderate", "deep"], index=1)
    style = st.selectbox("Output style", ["concise", "detailed", "technical"], index=1)
    
    st.divider()
    
    st.header("🎯 Advanced Features")
    use_agentic_rag = st.checkbox("Agentic RAG (self-questioning)", value=False)
    use_self_reflection = st.checkbox("Self-Reflective improvement", value=False)
    show_explanations = st.checkbox("Show retrieval explanations", value=True)
    
    st.divider()
    
    st.markdown("**System Stats:**")
    st.metric("Papers indexed", "200")
    st.metric("RL methods", "4")
    st.metric("AI agents", "9")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area(
        "🔍 Research Query",
        placeholder="What are recent advances in transformer models?",
        height=100
    )
    
    if st.button("🚀 Start Research", type="primary"):
        if query:
            with st.spinner("ARIA is researching..."):
                
                # Standard research
                result = st.session_state.orchestrator.research(
                    query,
                    {'top_k': top_k, 'depth': depth, 'style': style}
                )
                
                # Agentic RAG if enabled
                if use_agentic_rag:
                    with st.spinner("Using Agentic RAG..."):
                        agentic_result = st.session_state.agentic_rag.execute({
                            'query': query,
                            'max_subqueries': 3
                        })
                        st.session_state.agentic_result = agentic_result
                
                # Self-reflection if enabled
                if use_self_reflection:
                    with st.spinner("AI critiquing and improving..."):
                        reflected = st.session_state.reflective.execute({
                            'initial_summary': result['summary'],
                            'query': query,
                            'papers': result['key_papers']
                        })
                        result['summary'] = reflected['improved_summary']
                        result['reflection_data'] = reflected
                
                st.session_state.result = result
        else:
            st.error("Please enter a research query")

# Display results
if 'result' in st.session_state:
    result = st.session_state.result
    
    st.divider()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Quality Score", f"{result['quality_score']:.2f}")
    with col2:
        st.metric("Papers Analyzed", result['papers_analyzed'])
    with col3:
        st.metric("RL Reward", f"{result['reward']:.2f}")
    with col4:
        advanced = result['metadata'].get('advanced_prompts_used', False)
        st.metric("Advanced Prompts", "✓" if advanced else "✗")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Summary", "📚 Papers", "🔍 Explanations", "🤖 Agents"])
    
    with tab1:
        st.subheader("Research Summary")
        st.markdown(result['summary'])
        
        if 'reflection_data' in result:
            with st.expander("🪞 Self-Reflection Details"):
                refl = result['reflection_data']
                st.write(f"**Iterations:** {refl['improvement_iterations']}")
                st.write(f"**Quality:** {refl['initial_quality']} → {refl['final_quality']} (+{refl['improvement_percent']}%)")
                
                for i, critique in enumerate(refl['critiques'], 1):
                    st.write(f"**Iteration {i}:**")
                    st.write(f"- Quality: {critique['quality_score']}/100")
                    st.write(f"- Weaknesses: {', '.join(critique.get('weaknesses', []))}")
    
    with tab2:
        st.subheader(f"Key Papers ({len(result['key_papers'])})")
        for i, paper in enumerate(result['key_papers'], 1):
            st.write(f"{i}. {paper}")
        
        with st.expander("📑 All Citations"):
            for citation in result['citations']:
                st.text(citation)
    
    with tab3:
        if show_explanations and 'explanations' in st.session_state.result.get('execution_log', [{}])[1][1]:
            st.subheader("Why These Papers Were Selected")
            
            sources = [log[1] for log in result['execution_log'] if log[0] == 'source_discovery']
            if sources and 'explanations' in sources[0]:
                for exp in sources[0]['explanations'][:5]:
                    with st.expander(f"📄 #{exp['rank']}: {exp['paper_title'][:80]}..."):
                        st.write(f"**Overall Score:** {exp['overall_score']:.3f}")
                        st.write(f"**Why Selected:** {exp['why_selected']}")
                        st.write(f"**Key Factors:**")
                        for factor in exp['key_factors']:
                            st.write(f"  • {factor}")
        else:
            st.info("Enable 'Show retrieval explanations' in sidebar")
        
        if 'agentic_result' in st.session_state:
            st.divider()
            st.subheader("🤔 Agentic RAG - Self-Generated Questions")
            agentic = st.session_state.agentic_result
            for i, sq in enumerate(agentic['sub_questions'], 1):
                st.write(f"{i}. {sq}")
            st.metric("Unique papers found", agentic['unique_papers_count'])
    
    with tab4:
        st.subheader("Agent Execution Log")
        for agent_name, agent_data in result['execution_log']:
            with st.expander(f"🤖 {agent_name.replace('_', ' ').title()}"):
                st.json(agent_data, expanded=False)

# Footer
st.divider()
st.markdown("**ARIA** - Adaptive Research Intelligence Agent | Built with RL + GenAI")