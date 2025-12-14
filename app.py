# app.py
"""
ARIA Streamlit Web Demo - COMPLETE VERSION
Interactive research assistant with ALL GenAI features + New additions

NEW FEATURES:
- Citation Export (BibTeX/RIS/APA)
- Cost Tracking
- Confidence Scores
- Comparative Analysis
- Research Gap Identification
- Dark Mode
- Timeline Visualization
- ENHANCED: Structured summary display with expandable sections
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.tools.citation_exporter import CitationExporter
from src.utils.cost_tracker import CostTracker
from src.agents.comparative_analyzer import ComparativeAnalyzer
from src.agents.gap_identifier import ResearchGapIdentifier
from src.utils.confidence_extractor import ConfidenceExtractor
from src.tools.timeline_visualizer import TimelineVisualizer
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    with st.spinner("🚀 Loading ARIA system..."):
        st.session_state.vector_store = VectorStore()
        with open('data/papers/arxiv_papers.json', 'r') as f:
            papers = json.load(f)
        st.session_state.vector_store.add_papers(papers[:200])
        st.session_state.orchestrator = AgentOrchestrator(st.session_state.vector_store)
        st.session_state.agentic_rag = AgenticRAGAgent(st.session_state.vector_store)
        st.session_state.reflective = SelfReflectiveAgent()
        
        # NEW: Initialize all new tools
        st.session_state.citation_exporter = CitationExporter()
        st.session_state.cost_tracker = CostTracker()
        st.session_state.comparative_analyzer = ComparativeAnalyzer()
        st.session_state.gap_identifier = ResearchGapIdentifier()
        st.session_state.confidence_extractor = ConfidenceExtractor()
        st.session_state.timeline_viz = TimelineVisualizer()

# Sidebar
with st.sidebar:
    # Dark Mode Toggle
    st.markdown("### 🎨 Theme")
    dark_mode = st.toggle("🌙 Dark Mode", value=True)
    
    if dark_mode:
        st.markdown("""
        <style>
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            .stTextInput > div > div > input {
                background-color: #262730;
                color: #FAFAFA;
            }
            .stTextArea > div > div > textarea {
                background-color: #262730;
                color: #FAFAFA;
            }
            .stSelectbox > div > div > select {
                background-color: #262730;
                color: #FAFAFA;
            }
        </style>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.header("⚙️ Configuration")
    
    top_k = st.slider("Papers to retrieve", 3, 15, 12)
    depth = st.selectbox("Analysis depth", ["shallow", "moderate", "deep"], index=1)
    style = st.selectbox("Output style", ["concise", "detailed", "technical"], index=1)
    
    st.divider()
    
    st.header("🎯 Advanced Features")
    use_agentic_rag = st.checkbox("Agentic RAG (self-questioning)", value=False)
    use_self_reflection = st.checkbox("Self-Reflective improvement", value=False)
    show_explanations = st.checkbox("Show retrieval explanations", value=True)
    
    # NEW: Additional advanced features
    use_comparative = st.checkbox("Comparative Analysis (if applicable)", value=True)
    use_gap_finder = st.checkbox("Research Gap Identification", value=False)
    use_timeline = st.checkbox("Timeline Visualization", value=False)
    
    st.divider()
    
    st.markdown("**System Stats:**")
    st.metric("Papers indexed", "200")
    st.metric("RL methods", "4")
    st.metric("AI agents", "9")
    
    # NEW: Cost tracking display
    if 'result' in st.session_state:
        st.divider()
        st.markdown("**💰 Session Costs:**")
        cost_metrics = st.session_state.cost_tracker.get_cost_metrics()
        st.metric("Total Cost", cost_metrics['total_cost'])
        st.metric("Avg/Query", cost_metrics['avg_per_query'])
        st.metric("Queries", cost_metrics['queries'])

# Header
st.title("🔬 ARIA - Adaptive Research Intelligence Agent")
st.markdown("**AI research assistant with reinforcement learning + advanced GenAI features**")
st.caption("🆕 Now with Citation Export, Cost Tracking, Comparative Analysis, and more!")

# Main area
query = st.text_area(
    "🔍 Research Query",
    placeholder="What are recent advances in transformer models? OR Compare CNNs vs Transformers",
    height=100,
    key="main_query"
)

if st.button("🚀 Start Research", type="primary", use_container_width=True):
    if query:
        with st.spinner("ARIA is researching..."):
            
            # Standard research
            result = st.session_state.orchestrator.research(
                query,
                {'top_k': top_k, 'depth': depth, 'style': style}
            )
            
            # Track cost for base research
            st.session_state.cost_tracker.track_query(
                prompt=query,
                response=result['summary'],
                model='gpt-4o-mini'
            )
            
            # Agentic RAG if enabled
            if use_agentic_rag:
                with st.spinner("🤔 Using Agentic RAG..."):
                    agentic_result = st.session_state.agentic_rag.execute({
                        'query': query,
                        'max_subqueries': 3
                    })
                    st.session_state.agentic_result = agentic_result
                    
                    # Track cost
                    st.session_state.cost_tracker.track_query(
                        f"Agentic sub-questions for {query}",
                        str(agentic_result),
                        'gpt-4o-mini'
                    )
            
            # Self-reflection if enabled
            if use_self_reflection:
                with st.spinner("🪞 AI critiquing and improving..."):
                    reflected = st.session_state.reflective.execute({
                        'initial_summary': result['summary'],
                        'query': query,
                        'papers': result['key_papers']
                    })
                    result['summary'] = reflected['improved_summary']
                    result['reflection_data'] = reflected
                    
                    # Track cost
                    st.session_state.cost_tracker.track_query(
                        f"Self-reflection for {query}",
                        reflected['improved_summary'],
                        'gpt-4o-mini'
                    )
            
            # Comparative Analysis if enabled
            if use_comparative:
                with st.spinner("🔄 Checking for comparative analysis..."):
                    analysis_log = [log[1] for log in result['execution_log'] if log[0] == 'deep_reader']
                    if analysis_log:
                        analyzed_papers = analysis_log[0]['analyzed_papers']
                        
                        comparative_result = st.session_state.comparative_analyzer.execute({
                            'query': query,
                            'papers': result.get('execution_log', [{}])[1][1].get('papers', []),
                            'analyzed_papers': analyzed_papers
                        })
                        
                        if comparative_result.get('is_comparison', False):
                            st.session_state.comparative_result = comparative_result
                            st.success(f"✅ Comparative analysis: {comparative_result['entity1']} vs {comparative_result['entity2']}")
            
            # Research Gap Identification if enabled
            if use_gap_finder:
                with st.spinner("🔍 Identifying research gaps..."):
                    analysis_log = [log[1] for log in result['execution_log'] if log[0] == 'deep_reader']
                    query_log = [log[1] for log in result['execution_log'] if log[0] == 'query_analyzer']
                    
                    if analysis_log and query_log:
                        gap_result = st.session_state.gap_identifier.execute({
                            'query': query,
                            'papers': result.get('execution_log', [{}])[1][1].get('papers', []),
                            'analyzed_papers': analysis_log[0]['analyzed_papers'],
                            'domain': query_log[0]['domain']
                        })
                        st.session_state.gap_result = gap_result
                        st.success(f"✅ Found {len(gap_result.get('gaps_found', []))} potential research gaps")
            
            # Timeline Visualization if enabled
            if use_timeline:
                with st.spinner("📊 Creating timeline..."):
                    analysis_log = [log[1] for log in result['execution_log'] if log[0] == 'deep_reader']
                    if analysis_log:
                        timeline_result = st.session_state.timeline_viz.create_timeline(
                            analysis_log[0]['analyzed_papers']
                        )
                        st.session_state.timeline_result = timeline_result
                        st.success("✅ Research timeline created")
            
            st.session_state.result = result
            st.success("✅ Research complete!")
    else:
        st.error("⚠️ Please enter a research query")

# Display results
if 'result' in st.session_state:
    result = st.session_state.result
    
    st.divider()
    
    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Quality Score", f"{result['quality_score']:.2f}")
    with col2:
        st.metric("Papers Analyzed", result['papers_analyzed'])
    with col3:
        st.metric("RL Reward", f"{result['reward']:.2f}")
    with col4:
        advanced = result['metadata'].get('advanced_prompts_used', False)
        st.metric("Advanced Prompts", "✓" if advanced else "✗")
    with col5:
        cost_summary = st.session_state.cost_tracker.get_session_summary()
        last_cost = st.session_state.cost_tracker.session_costs[-1]['cost'] if st.session_state.cost_tracker.session_costs else 0
        st.metric("Query Cost", st.session_state.cost_tracker.format_cost(last_cost))
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Summary", 
        "📚 Papers & Citations", 
        "🔍 Explanations", 
        "🔄 Advanced Analysis",
        "💰 Cost Breakdown",
        "🤖 Agents"
    ])
    
    # TAB 1: Summary
    with tab1:
        st.subheader("📄 Research Summary")
        
        summary_text = result['summary']
        sections = summary_text.split('##')
        
        if len(sections) > 1:
            for section in sections[1:]:
                lines = section.strip().split('\n', 1)
                if len(lines) >= 1:
                    section_title = lines[0].strip()
                    section_content = lines[1].strip() if len(lines) > 1 else ""
                    
                    with st.expander(section_title, expanded=True):
                        if section_content:
                            st.markdown(section_content)
        else:
            st.markdown(summary_text)
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"📚 Based on {result['papers_analyzed']} papers")
        with col2:
            query_analysis = result.get('query_analysis', {})
            st.caption(f"🏷️ Domain: {query_analysis.get('domain', 'general').upper()}")
        with col3:
            st.caption(f"⏱️ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        if 'reflection_data' in result:
            st.divider()
            st.subheader("🪞 Self-Reflection Analysis")
            
            refl = result['reflection_data']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Iterations", refl['improvement_iterations'])
            with col2:
                st.metric("Improvement", f"+{refl['improvement_percent']}%")
            
            st.write(f"**Quality:** {refl['initial_quality']} → {refl['final_quality']}")
            
            st.write("**Iteration Details:**")
            for i, critique in enumerate(refl['critiques'], 1):
                st.write(f"**Iteration {i}:** Quality {critique['quality_score']}/100")
                if critique.get('weaknesses'):
                    st.write(f"  - Weaknesses: {', '.join(critique['weaknesses'][:2])}")
    
    # TAB 2: Papers & Citations
    with tab2:
        st.subheader(f"📚 Key Papers ({len(result['key_papers'])})")
        
        for i, paper in enumerate(result['key_papers'], 1):
            st.write(f"{i}. {paper}")
        
        st.divider()
        st.subheader("📥 Export Citations")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            citation_format = st.selectbox("Format:", ["BibTeX", "RIS", "APA"], key="citation_format")
        
        with col2:
            if st.button("📋 Generate Citations", key="gen_citations_btn", use_container_width=True):
                sources = [log[1] for log in result['execution_log'] if log[0] == 'source_discovery']
                if sources:
                    papers_to_export = sources[0]['papers'][:result['papers_analyzed']]
                    format_map = {'BibTeX': 'bibtex', 'RIS': 'ris', 'APA': 'apa'}
                    citations = st.session_state.citation_exporter.export_batch(
                        papers_to_export, 
                        format_map[citation_format]
                    )
                    st.session_state.generated_citations = citations
                    st.session_state.citation_format_used = citation_format
        
        if 'generated_citations' in st.session_state:
            st.text_area(
                f"📑 {st.session_state.citation_format_used} Citations",
                st.session_state.generated_citations,
                height=300
            )
            
            st.download_button(
                label=f"💾 Download {st.session_state.citation_format_used}",
                data=st.session_state.generated_citations,
                file_name=f"aria_citations_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        if 'timeline_result' in st.session_state and use_timeline:
            st.divider()
            st.subheader("📊 Research Timeline")
            
            timeline = st.session_state.timeline_result
            
            if 'figure_path' in timeline and timeline['figure_path']:
                st.image(timeline['figure_path'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Year Range:** {timeline.get('year_range', 'N/A')}")
            with col2:
                st.write(f"**Papers:** {timeline.get('total_papers', 0)}")
    
    # TAB 3: Explanations
    with tab3:
        if show_explanations:
            sources = [log[1] for log in result['execution_log'] if log[0] == 'source_discovery']
            if sources and 'explanations' in sources[0]:
                st.subheader("🔍 Paper Selection Explanations")
                
                for exp in sources[0]['explanations'][:5]:
                    with st.expander(f"📄 #{exp['rank']}: {exp['paper_title'][:60]}..."):
                        st.write(f"**Score:** {exp['overall_score']:.3f}")
                        st.write(f"**Why:** {exp['why_selected']}")
                        st.write("**Factors:**")
                        for factor in exp['key_factors']:
                            st.write(f"• {factor}")
        
        if 'agentic_result' in st.session_state:
            st.divider()
            st.subheader("🤔 Agentic RAG Questions")
            agentic = st.session_state.agentic_result
            
            for i, sq in enumerate(agentic['sub_questions'], 1):
                st.info(f"**Q{i}:** {sq}")
            
            st.metric("Unique papers", agentic['unique_papers_count'])
    
    # TAB 4: Advanced Analysis
    with tab4:
        if 'comparative_result' in st.session_state:
            comp = st.session_state.comparative_result
            
            st.subheader(f"🔄 {comp['entity1'].title()} vs {comp['entity2'].title()}")
            
            if comp.get('comparison_table'):
                st.write("### Comparison")
                table_data = []
                for aspect, values in comp['comparison_table'].items():
                    table_data.append({
                        'Aspect': aspect,
                        comp['entity1'].title(): values.get(comp['entity1'], values.get(comp['entity1'].title(), 'N/A')),
                        comp['entity2'].title(): values.get(comp['entity2'], values.get(comp['entity2'].title(), 'N/A'))
                    })
                st.table(table_data)
            
            if comp.get('narrative'):
                st.divider()
                st.markdown(comp['narrative'])
            
            if comp.get('when_to_use'):
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Use {comp['entity1'].title()}:**")
                    for uc in comp['when_to_use'].get(comp['entity1'], comp['when_to_use'].get(comp['entity1'].title(), [])):
                        st.write(f"- {uc}")
                with col2:
                    st.write(f"**Use {comp['entity2'].title()}:**")
                    for uc in comp['when_to_use'].get(comp['entity2'], comp['when_to_use'].get(comp['entity2'].title(), [])):
                        st.write(f"- {uc}")
        
        if 'gap_result' in st.session_state:
            if 'comparative_result' in st.session_state:
                st.divider()
            
            st.subheader("🔍 Research Gaps")
            
            gaps = st.session_state.gap_result
            
            if gaps.get('gaps_found'):
                for i, gap in enumerate(gaps['gaps_found'], 1):
                    conf_icon = {'high': '🟢', 'medium': '🟡', 'low': '🟠'}.get(gap.get('confidence', 'medium').lower(), '⚪')
                    
                    with st.expander(f"{conf_icon} Gap {i}: {gap.get('gap', '')[:60]}...", expanded=(i==1)):
                        st.write(f"**Confidence:** {gap.get('confidence', 'medium').title()}")
                        st.write(f"**Why:** {gap.get('rationale', 'N/A')}")
            
            if gaps.get('suggested_directions'):
                st.divider()
                st.write("**💡 Future Directions:**")
                for d in gaps['suggested_directions']:
                    st.write(f"- {d}")
        
        if not ('comparative_result' in st.session_state or 'gap_result' in st.session_state):
            st.info("💡 Enable Comparative Analysis or Research Gap Identification in sidebar")
    
    # TAB 5: Cost Breakdown
    with tab5:
        st.subheader("💰 Cost Analysis")
        
        cost_summary = st.session_state.cost_tracker.get_session_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", st.session_state.cost_tracker.format_cost(cost_summary['total_cost']))
        with col2:
            st.metric("Queries", cost_summary['total_queries'])
        with col3:
            st.metric("Avg/Query", st.session_state.cost_tracker.format_cost(cost_summary['avg_cost_per_query']))
        with col4:
            st.metric("Tokens", f"{cost_summary['total_tokens']:,}")
        
        st.divider()
        st.write("**Token Breakdown:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input", f"{cost_summary['total_input_tokens']:,}")
        with col2:
            st.metric("Output", f"{cost_summary['total_output_tokens']:,}")
        with col3:
            st.metric("Avg/Query", f"{cost_summary['avg_tokens_per_query']:.0f}")
        
        st.divider()
        remaining = 10.0 - cost_summary['total_cost']
        queries_left = int(remaining / max(cost_summary['avg_cost_per_query'], 0.001))
        
        st.info(f"💡 Budget: {st.session_state.cost_tracker.format_cost(cost_summary['total_cost'])}/$ 10.00 used | ~{queries_left} queries remaining")
    
    # TAB 6: Agents
    with tab6:
        st.subheader("🤖 Execution Pipeline")
        st.write(f"**Agents:** {len(result['execution_log'])}")
        
        st.divider()
        
        for i, (agent_name, agent_data) in enumerate(result['execution_log'], 1):
            with st.expander(f"{i}. {agent_name.replace('_', ' ').title()}"):
                if agent_name == 'query_analyzer':
                    st.write(f"Domain: {agent_data.get('domain', 'N/A')} | Complexity: {agent_data.get('complexity', 'N/A')}")
                elif agent_name == 'source_discovery':
                    st.write(f"Papers: {agent_data.get('count', 0)} | Credibility: {agent_data.get('avg_credibility', 0):.2f}")
                elif agent_name == 'quality_evaluator':
                    metrics = agent_data.get('metrics', {})
                    st.write(f"Completeness: {metrics.get('completeness', 0):.2f} | Depth: {metrics.get('depth', 0):.2f}")
                else:
                    st.json(agent_data, expanded=False)

# Footer
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown("**ARIA** - Adaptive Research Intelligence Agent")
    st.caption("v2.0 | Northeastern University")

with footer_col2:
    if st.button("🔄 Reset Costs"):
        st.session_state.cost_tracker.reset_session()
        st.rerun()

with footer_col3:
    features_enabled = sum([use_agentic_rag, use_self_reflection, show_explanations, use_comparative, use_gap_finder, use_timeline])
    st.metric("Features", f"{features_enabled}/6")