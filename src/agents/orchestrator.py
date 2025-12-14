# src/agents/orchestrator.py
"""
Production-Grade Agent Orchestrator for ARIA
ALL GenAI features integrated: Tier 1 + Tier 2 + Tier 3
FIXED: Passes top_k to Deep Reader for correct paper count
"""

from typing import Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.query_analyzer import QueryAnalyzerAgent
from src.agents.source_discovery import SourceDiscoveryAgent
from src.agents.deep_reader import DeepReaderAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.quality_evaluator import QualityEvaluatorAgent
from src.agents.citation_analyzer import CitationAnalyzerAgent
from src.agents.contradiction_detector import ContradictionDetectorAgent
from src.agents.trend_forecaster import TrendForecasterAgent
from src.agents.methodology_scout import MethodologyScoutAgent
from src.rag.vector_store import VectorStore
from src.utils.validators import validate_query, validate_config, validate_papers
from src.utils.logger import get_logger
from src.utils.metrics_tracker import get_performance_tracker
from src.utils.error_handler import safe_execute, AgentExecutionError

# Import Tier 3 features
try:
    from src.tools.knowledge_graph import KnowledgeGraph
    from src.tools.arxiv_live import ArxivLiveIntegration
    from src.agents.adaptive_summarizer import AdaptiveSummarizer
    TIER3_AVAILABLE = True
except ImportError:
    TIER3_AVAILABLE = False

class AgentOrchestrator:
    """
    Production-grade multi-agent orchestrator
    Integrates ALL GenAI innovations
    """
    
    def __init__(self, vector_store: VectorStore, enable_tier3: bool = False):
        """Initialize with optional Tier 3 features"""
        
        # Core agents
        self.query_analyzer = QueryAnalyzerAgent()
        self.source_discovery = SourceDiscoveryAgent(vector_store)
        self.deep_reader = DeepReaderAgent()
        self.synthesizer = SynthesizerAgent()
        self.quality_evaluator = QualityEvaluatorAgent()
        self.citation_analyzer = CitationAnalyzerAgent()
        self.contradiction_detector = ContradictionDetectorAgent()
        self.trend_forecaster = TrendForecasterAgent()
        self.methodology_scout = MethodologyScoutAgent()
        
        self.execution_log = []
        
        # Monitoring
        self.logger = get_logger()
        self.perf_tracker = get_performance_tracker()
        
        # Tier 3 features (optional)
        self.enable_tier3 = enable_tier3 and TIER3_AVAILABLE
        if self.enable_tier3:
            self.knowledge_graph = KnowledgeGraph()
            self.arxiv_live = ArxivLiveIntegration()
            self.adaptive_summarizer = AdaptiveSummarizer()
            print("   ✓ Tier 3 features enabled")
        
        self.logger.logger.info(f"AgentOrchestrator initialized (Tier 3: {self.enable_tier3})")
    
    def research(self, query: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute complete research pipeline"""
        
        operation_start = self.perf_tracker.start_operation()
        
        try:
            # Validation
            query = validate_query(query)
            if config is None:
                config = {}
            config = validate_config(config)
            
            self.logger.log_research_start(query, config)
            
            top_k = config.get('top_k', 10)
            depth = config.get('depth', 'moderate')
            style = config.get('style', 'detailed')
            user_level = config.get('user_level', 'phd')
            use_live_arxiv = config.get('use_live_arxiv', False)
            
            print(f"\n{'='*60}")
            print(f"🔬 Starting Research: {query}")
            print(f"{'='*60}\n")
            
            # Step 1: Query Analysis
            print("1️⃣  Query Analyzer working...")
            try:
                query_analysis = self.query_analyzer.execute({'query': query})
                self.execution_log.append(('query_analyzer', query_analysis))
                self.logger.log_agent_execution('query_analyzer', {'query': query}, query_analysis)
                
                print(f"   ✓ Domain: {query_analysis['domain']}")
                print(f"   ✓ Complexity: {query_analysis['complexity']}")
                print(f"   ✓ Type: {query_analysis['query_type']}")
            except Exception as e:
                self.logger.log_error('query_analyzer', e, {'query': query})
                query_analysis = {
                    'query': query,
                    'keywords': query.lower().split()[:5],
                    'domain': 'general',
                    'complexity': 'moderate',
                    'query_type': 'exploratory',
                    'agent': 'Query Analyzer (fallback)'
                }
                print(f"   ⚠️  Fallback query analysis")
            
            # Step 2: Source Discovery
            print("\n2️⃣  Source Discovery working...")
            
            arxiv_papers = []
            if self.enable_tier3 and use_live_arxiv:
                print("   🔴 Checking arXiv for papers from last 7 days...")
                try:
                    arxiv_papers = self.arxiv_live.get_latest_papers(
                        query=query,
                        days_back=7,
                        max_results=3
                    )
                    print(f"   ✓ Found {len(arxiv_papers)} new papers from arXiv")
                except:
                    print("   ⚠️  arXiv check skipped")
            
            try:
                sources = self.source_discovery.execute({
                    'query': query,
                    'keywords': query_analysis['keywords'],
                    'domain': query_analysis['domain'],
                    'top_k': top_k,
                    'explain': True
                })
                
                sources['papers'] = validate_papers(sources['papers'])
                
                if arxiv_papers:
                    sources['live_arxiv_papers'] = arxiv_papers
                    sources['has_live_papers'] = True
                
                self.execution_log.append(('source_discovery', sources))
                self.logger.log_agent_execution('source_discovery', query_analysis, sources)
                
                print(f"   ✓ Found {sources['count']} papers")
                print(f"   ✓ Avg Credibility: {sources['avg_credibility']:.2f}")
                
                if sources['count'] == 0:
                    raise AgentExecutionError("No papers found")
                    
            except Exception as e:
                self.logger.log_error('source_discovery', e, {'query': query})
                raise AgentExecutionError(f"Source discovery failed: {e}")
            
            # Build knowledge graph (Tier 3)
            if self.enable_tier3:
                print("\n🕸️  Building knowledge graph...")
                try:
                    self.knowledge_graph.build_from_papers(sources['papers'])
                    sources['knowledge_graph'] = self.knowledge_graph.get_stats()
                except:
                    print("   ⚠️  Knowledge graph skipped")
            
            # Step 3: Deep Analysis (FIXED - PASSES num_papers)
            print("\n3️⃣  Deep Reader analyzing...")
            try:
                analysis = self.deep_reader.execute({
                    'papers': sources['papers'],
                    'depth': depth,
                    'query': query,
                    'domain': query_analysis['domain'],
                    'query_type': query_analysis['query_type'],
                    'num_papers': top_k  # FIXED: Now passes top_k to analyze all retrieved papers
                })
                self.execution_log.append(('deep_reader', analysis))
                
                print(f"   ✓ Analyzed {analysis['papers_analyzed']} papers")
                print(f"   ✓ {len(analysis['key_insights'])} char insights")
                print(f"   ✓ {analysis.get('prompt_type', 'standard')} prompts")
            except Exception as e:
                self.logger.log_error('deep_reader', e)
                analysis = {
                    'analyzed_papers': [
                        {'title': p['metadata']['title'],
                         'relevance_score': p.get('score', 0),
                         'credibility_score': p.get('credibility', {}).get('credibility_score', 0),
                         'domain': p['metadata']['domain'],
                         'published': p['metadata']['published']}
                        for p in sources['papers'][:top_k]
                    ],
                    'key_insights': 'Fallback analysis',
                    'papers_analyzed': min(top_k, sources['count']),
                    'depth': depth
                }
                print(f"   ⚠️  Fallback")
            
            # Step 4: Specialized Analysis
            print("\n4️⃣  Specialized analysis...")
            
            citation_results = safe_execute(
                lambda: self.citation_analyzer.execute({'papers': sources['papers']}),
                fallback_value={'high_impact_papers': [], 'citation_clusters': []},
                log_error=True
            )
            self.execution_log.append(('citation_analyzer', citation_results))
            print(f"   ✓ Citations: {len(citation_results['high_impact_papers'])}")
            
            contradiction_results = safe_execute(
                lambda: self.contradiction_detector.execute({'analyzed_papers': analysis['analyzed_papers']}),
                fallback_value={'contradictions': [], 'controversial_topics': []},
                log_error=True
            )
            self.execution_log.append(('contradiction_detector', contradiction_results))
            
            trend_results = safe_execute(
                lambda: self.trend_forecaster.execute({'papers': sources['papers']}),
                fallback_value={'emerging_trends': [], 'hot_topics': []},
                log_error=True
            )
            self.execution_log.append(('trend_forecaster', trend_results))
            
            methodology_results = safe_execute(
                lambda: self.methodology_scout.execute({'papers': sources['papers']}),
                fallback_value={'methodologies': [], 'common_approaches': []},
                log_error=True
            )
            self.execution_log.append(('methodology_scout', methodology_results))
            
            # Step 5: Synthesis
            print("\n5️⃣  Synthesizing...")
            try:
                synthesis = self.synthesizer.execute({
                    'query': query,
                    'analyzed_papers': analysis['analyzed_papers'],
                    'key_insights': analysis['key_insights'],
                    'style': style,
                    'domain': query_analysis['domain']
                })
                self.execution_log.append(('synthesizer', synthesis))
                print(f"   ✓ {len(synthesis['summary'])} chars")
                print(f"   ✓ {synthesis.get('prompt_type', 'standard')} prompts")
            except Exception as e:
                self.logger.log_error('synthesizer', e)
                synthesis = {
                    'summary': self._generate_fallback_summary(query, analysis['analyzed_papers']),
                    'key_papers': [p['title'] for p in analysis['analyzed_papers'][:5]],
                    'citations': [f"{p['title']} ({p['published'][:4]})" for p in analysis['analyzed_papers'][:5]],
                    'style': style
                }
                print(f"   ⚠️  Fallback")
            
            # Adaptive summarization (Tier 3)
            if self.enable_tier3 and user_level != 'phd':
                print(f"\n🎯 Adapting for {user_level} audience...")
                try:
                    adapted = self.adaptive_summarizer.execute({
                        'summary': synthesis['summary'],
                        'query': query,
                        'user_level': user_level
                    })
                    synthesis['summary'] = adapted['adapted_summary']
                    synthesis['adapted_for'] = user_level
                    print(f"   ✓ Adapted for {user_level}")
                except:
                    print(f"   ⚠️  Adaptation skipped")
            
            # Step 6: Quality Evaluation
            print("\n6️⃣  Quality evaluation...")
            try:
                evaluation = self.quality_evaluator.execute({
                    'summary': synthesis['summary'],
                    'papers_count': sources['count'],
                    'citations': synthesis['citations'],
                    'query_complexity': query_analysis['complexity']
                })
                self.execution_log.append(('quality_evaluator', evaluation))
                print(f"   ✓ Quality: {evaluation['quality_score']:.2f}")
                print(f"   ✓ Reward: {evaluation['reward']:.2f}")
            except Exception as e:
                self.logger.log_error('quality_evaluator', e)
                evaluation = {
                    'quality_score': 0.5,
                    'metrics': {'completeness': 0.5, 'depth': 0.5, 'coherence': 0.5, 'citation_quality': 0.5},
                    'reward': 0.0
                }
                print(f"   ⚠️  Fallback")
            
            # Compile results
            results = {
                'query': query,
                'query_analysis': query_analysis,
                'papers_found': sources['count'],
                'papers_analyzed': analysis['papers_analyzed'],
                'summary': synthesis['summary'],
                'key_papers': synthesis['key_papers'],
                'citations': synthesis['citations'],
                'quality_metrics': evaluation['metrics'],
                'quality_score': evaluation['quality_score'],
                'reward': evaluation['reward'],
                'additional_analysis': {
                    'citation_analysis': citation_results,
                    'contradictions': contradiction_results,
                    'trends': trend_results,
                    'methodologies': methodology_results
                },
                'execution_log': self.execution_log.copy(),
                'metadata': {
                    'config_used': config,
                    'fallback_used': 'fallback' in str(synthesis.get('agent', '')),
                    'agents_executed': len(self.execution_log),
                    'advanced_prompts_used': analysis.get('prompt_type') == 'advanced_cot_fewshot',
                    'tier3_features': {
                        'knowledge_graph': self.enable_tier3,
                        'live_arxiv': sources.get('has_live_papers', False),
                        'adaptive_summary': synthesis.get('adapted_for') is not None
                    }
                }
            }
            
            # Add Tier 3 data if available
            if self.enable_tier3:
                if 'knowledge_graph' in sources:
                    results['knowledge_graph_stats'] = sources['knowledge_graph']
                if 'live_arxiv_papers' in sources:
                    results['live_arxiv_papers'] = sources['live_arxiv_papers']
            
            print(f"\n{'='*60}")
            print(f"✅ Research Complete!")
            print(f"{'='*60}\n")
            
            self.perf_tracker.end_operation(operation_start, 'full_research')
            
            self.logger.log_performance_metrics({
                'quality_score': evaluation['quality_score'],
                'papers_analyzed': analysis['papers_analyzed'],
                'agents_called': len(self.execution_log),
                'tier3_enabled': self.enable_tier3
            })
            
            self.execution_log = []
            
            return results
            
        except Exception as e:
            self.logger.log_error('orchestrator', e, {'query': query, 'config': config})
            self.perf_tracker.end_operation(operation_start, 'failed')
            raise AgentExecutionError(f"Pipeline failed: {e}") from e
    
    def _generate_fallback_summary(self, query: str, papers: List[Dict]) -> str:
        """Generate fallback summary"""
        summary = f"Research Summary: {query}\n\nAnalyzed {len(papers)} papers.\n\nKey Papers:\n"
        for i, p in enumerate(papers[:5], 1):
            summary += f"{i}. {p['title']} ({p['published'][:4]})\n"
        return summary
    
    def get_state(self) -> Dict[str, Any]:
        """Get system state"""
        return {
            'last_execution_log': self.execution_log,
            'agent_memories': {
                'query_analyzer': len(self.query_analyzer.memory),
                'source_discovery': len(self.source_discovery.memory),
                'deep_reader': len(self.deep_reader.memory),
                'synthesizer': len(self.synthesizer.memory),
                'quality_evaluator': len(self.quality_evaluator.memory)
            },
            'performance_metrics': self.perf_tracker.get_metrics(),
            'tier3_enabled': self.enable_tier3
        }
    
    def get_performance_summary(self):
        """Print performance"""
        self.perf_tracker.print_summary()
    
    def save_logs(self, output_path: str = None):
        """Save logs"""
        return self.logger.save_session_log(output_path)