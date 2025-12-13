# src/agents/orchestrator.py
"""
Production-Grade Agent Orchestrator for ARIA
Includes comprehensive error handling, logging, and performance monitoring
"""

from typing import Dict, Any, List
import sys
import os

# Add project root to path
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

class AgentOrchestrator:
    """
    Production-grade multi-agent orchestrator with comprehensive error handling
    
    Features:
    - 9 specialized agents for research tasks
    - Advanced prompt engineering integration
    - Input validation and sanitization
    - Performance monitoring and logging
    - Graceful error handling with fallbacks
    - Execution tracking for RL training
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize agent orchestrator
        
        Args:
            vector_store: Vector store for paper retrieval
        """
        # Initialize all agents
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
        
        # Initialize monitoring
        self.logger = get_logger()
        self.perf_tracker = get_performance_tracker()
        
        self.logger.logger.info("AgentOrchestrator initialized with 9 agents")
    
    def research(self, query: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute complete research pipeline with production-grade error handling
        
        Args:
            query: Research question
            config: Configuration for research process
                - top_k: Number of papers to retrieve (default: 10)
                - depth: Analysis depth ('shallow', 'moderate', 'deep')
                - style: Output style ('concise', 'detailed', 'technical')
        
        Returns:
            Complete research results with quality metrics
        
        Raises:
            ValidationError: If inputs are invalid
            AgentExecutionError: If critical agent fails
        """
        # Start performance tracking
        operation_start = self.perf_tracker.start_operation()
        
        try:
            # Input validation
            query = validate_query(query)
            
            if config is None:
                config = {}
            
            config = validate_config(config)
            
            # Log research start
            self.logger.log_research_start(query, config)
            
            # Extract config
            top_k = config.get('top_k', 10)
            depth = config.get('depth', 'moderate')
            style = config.get('style', 'detailed')
            
            print(f"\n{'='*60}")
            print(f"🔬 Starting Research: {query}")
            print(f"{'='*60}\n")
            
            # Step 1: Analyze Query (Critical - must succeed)
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
                # Provide fallback analysis
                query_analysis = {
                    'query': query,
                    'keywords': query.lower().split()[:5],
                    'domain': 'general',
                    'complexity': 'moderate',
                    'query_type': 'exploratory',
                    'agent': 'Query Analyzer (fallback)'
                }
                print(f"   ⚠️  Using fallback query analysis")
            
            # Step 2: Discover Sources (Critical - must succeed)
            print("\n2️⃣  Source Discovery working...")
            try:
                sources = self.source_discovery.execute({
                    'query': query,
                    'keywords': query_analysis['keywords'],
                    'domain': query_analysis['domain'],
                    'top_k': top_k
                })
                
                # Validate papers
                sources['papers'] = validate_papers(sources['papers'])
                
                self.execution_log.append(('source_discovery', sources))
                self.logger.log_agent_execution('source_discovery', query_analysis, sources)
                
                print(f"   ✓ Found {sources['count']} papers")
                print(f"   ✓ Avg Credibility: {sources['avg_credibility']:.2f} (via {sources['credibility_tool_used']})")
                
                if sources['count'] == 0:
                    raise AgentExecutionError("No papers found for query")
                    
            except Exception as e:
                self.logger.log_error('source_discovery', e, {'query': query})
                raise AgentExecutionError(f"Source discovery failed: {e}")
            
            # Step 3: Analyze Papers with Advanced Prompts
            print("\n3️⃣  Deep Reader analyzing papers...")
            try:
                analysis = self.deep_reader.execute({
                    'papers': sources['papers'],
                    'depth': depth,
                    'query': query,
                    'domain': query_analysis['domain'],          # ADDED
                    'query_type': query_analysis['query_type']   # ADDED
                })
                self.execution_log.append(('deep_reader', analysis))
                self.logger.log_agent_execution('deep_reader', sources, analysis)
                
                print(f"   ✓ Analyzed {analysis['papers_analyzed']} papers")
                print(f"   ✓ Extracted {len(analysis['key_insights'])} character insights")
                print(f"   ✓ Using: {analysis.get('prompt_type', 'standard')} prompts")
            except Exception as e:
                self.logger.log_error('deep_reader', e)
                # Fallback: basic analysis
                analysis = {
                    'analyzed_papers': [
                        {
                            'title': p['metadata']['title'],
                            'relevance_score': p.get('score', 0),
                            'credibility_score': p.get('credibility', {}).get('credibility_score', 0),
                            'domain': p['metadata']['domain'],
                            'published': p['metadata']['published']
                        }
                        for p in sources['papers'][:min(5, len(sources['papers']))]
                    ],
                    'key_insights': 'Analysis performed using fallback method.',
                    'papers_analyzed': min(5, sources['count']),
                    'depth': depth,
                    'agent': 'Deep Reader (fallback)'
                }
                print(f"   ⚠️  Using fallback analysis")
            
            # Step 4: Additional Analysis (Optional - failures allowed)
            print("\n4️⃣  Running specialized analysis...")
            
            citation_results = safe_execute(
                lambda: self.citation_analyzer.execute({'papers': sources['papers']}),
                fallback_value={'high_impact_papers': [], 'citation_clusters': [], 'agent': 'Citation Analyzer (skipped)'},
                log_error=True
            )
            self.execution_log.append(('citation_analyzer', citation_results))
            print(f"   ✓ Citation Analyzer: {len(citation_results['high_impact_papers'])} high-impact papers")
            
            contradiction_results = safe_execute(
                lambda: self.contradiction_detector.execute({'analyzed_papers': analysis['analyzed_papers']}),
                fallback_value={'contradictions': [], 'controversial_topics': [], 'agent': 'Contradiction Detector (skipped)'},
                log_error=True
            )
            self.execution_log.append(('contradiction_detector', contradiction_results))
            print(f"   ✓ Contradiction Detector: {len(contradiction_results['contradictions'])} potential conflicts")
            
            trend_results = safe_execute(
                lambda: self.trend_forecaster.execute({'papers': sources['papers']}),
                fallback_value={'emerging_trends': [], 'hot_topics': [], 'agent': 'Trend Forecaster (skipped)'},
                log_error=True
            )
            self.execution_log.append(('trend_forecaster', trend_results))
            print(f"   ✓ Trend Forecaster: {len(trend_results['emerging_trends'])} trends identified")
            
            methodology_results = safe_execute(
                lambda: self.methodology_scout.execute({'papers': sources['papers']}),
                fallback_value={'methodologies': [], 'common_approaches': [], 'agent': 'Methodology Scout (skipped)'},
                log_error=True
            )
            self.execution_log.append(('methodology_scout', methodology_results))
            print(f"   ✓ Methodology Scout: {len(methodology_results['methodologies'])} methods found")
            
            # Step 5: Synthesize with Advanced Prompts
            print("\n5️⃣  Synthesizer creating summary...")
            try:
                synthesis = self.synthesizer.execute({
                    'query': query,
                    'analyzed_papers': analysis['analyzed_papers'],
                    'key_insights': analysis['key_insights'],
                    'style': style,
                    'domain': query_analysis['domain']  # ADDED
                })
                self.execution_log.append(('synthesizer', synthesis))
                print(f"   ✓ Generated {len(synthesis['summary'])} character summary")
                print(f"   ✓ Using: {synthesis.get('prompt_type', 'standard')} prompts")
            except Exception as e:
                self.logger.log_error('synthesizer', e)
                synthesis = {
                    'summary': self._generate_fallback_summary(query, analysis['analyzed_papers']),
                    'key_papers': [p['title'] for p in analysis['analyzed_papers'][:5]],
                    'citations': [f"{p['title']} ({p['published'][:4]})" for p in analysis['analyzed_papers'][:5]],
                    'style': style,
                    'agent': 'Synthesizer (fallback)'
                }
                print(f"   ⚠️  Using fallback synthesis")
            
            # Step 6: Evaluate Quality
            print("\n6️⃣  Quality Evaluator assessing output...")
            try:
                evaluation = self.quality_evaluator.execute({
                    'summary': synthesis['summary'],
                    'papers_count': sources['count'],
                    'citations': synthesis['citations'],
                    'query_complexity': query_analysis['complexity']
                })
                self.execution_log.append(('quality_evaluator', evaluation))
                print(f"   ✓ Quality Score: {evaluation['quality_score']:.2f}")
                print(f"   ✓ RL Reward: {evaluation['reward']:.2f}")
            except Exception as e:
                self.logger.log_error('quality_evaluator', e)
                evaluation = {
                    'quality_score': 0.5,
                    'metrics': {'completeness': 0.5, 'depth': 0.5, 'coherence': 0.5, 'citation_quality': 0.5},
                    'reward': 0.0,
                    'agent': 'Quality Evaluator (fallback)'
                }
                print(f"   ⚠️  Using fallback evaluation")
            
            # Compile final results
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
                    'fallback_used': 'fallback' in str(synthesis.get('agent', '')) or 'fallback' in str(evaluation.get('agent', '')),
                    'agents_executed': len(self.execution_log),
                    'advanced_prompts_used': analysis.get('prompt_type') == 'advanced_cot_fewshot'
                }
            }
            
            print(f"\n{'='*60}")
            print(f"✅ Research Complete!")
            print(f"{'='*60}\n")
            
            # End performance tracking
            self.perf_tracker.end_operation(operation_start, 'full_research')
            
            # Log performance metrics
            self.logger.log_performance_metrics({
                'quality_score': evaluation['quality_score'],
                'papers_analyzed': analysis['papers_analyzed'],
                'agents_called': len(self.execution_log),
                'fallback_used': results['metadata']['fallback_used'],
                'advanced_prompts': results['metadata']['advanced_prompts_used']
            })
            
            # Reset log for next query
            self.execution_log = []
            
            return results
            
        except Exception as e:
            # Log critical failure
            self.logger.log_error('orchestrator', e, {'query': query, 'config': config})
            
            # End tracking even on failure
            self.perf_tracker.end_operation(operation_start, 'full_research_failed')
            
            # Re-raise with context
            raise AgentExecutionError(f"Research pipeline failed: {e}") from e
    
    def _generate_fallback_summary(self, query: str, papers: List[Dict]) -> str:
        """Generate fallback summary when LLM unavailable"""
        summary = f"Research Summary: {query}\n\n"
        summary += f"Analyzed {len(papers)} relevant papers from recent research.\n\n"
        summary += "Key Papers:\n"
        
        for i, paper in enumerate(papers[:5], 1):
            summary += f"{i}. {paper['title']} ({paper['published'][:4]})\n"
            summary += f"   Relevance: {paper.get('relevance_score', 0):.2f} | "
            summary += f"Credibility: {paper.get('credibility_score', 0):.2f}\n"
        
        summary += f"\nBased on analysis of {len(papers)} papers across multiple research domains."
        
        return summary
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the system (for RL)"""
        return {
            'last_execution_log': self.execution_log,
            'agent_memories': {
                'query_analyzer': len(self.query_analyzer.memory),
                'source_discovery': len(self.source_discovery.memory),
                'deep_reader': len(self.deep_reader.memory),
                'synthesizer': len(self.synthesizer.memory),
                'quality_evaluator': len(self.quality_evaluator.memory),
                'citation_analyzer': len(self.citation_analyzer.memory),
                'contradiction_detector': len(self.contradiction_detector.memory),
                'trend_forecaster': len(self.trend_forecaster.memory),
                'methodology_scout': len(self.methodology_scout.memory)
            },
            'performance_metrics': self.perf_tracker.get_metrics()
        }
    
    def get_performance_summary(self):
        """Print performance summary"""
        self.perf_tracker.print_summary()
    
    def save_logs(self, output_path: str = None):
        """Save session logs"""
        return self.logger.save_session_log(output_path)