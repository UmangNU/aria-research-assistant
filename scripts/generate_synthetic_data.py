# scripts/generate_synthetic_data.py
"""
Synthetic Data Generation for ARIA - MAXIMUM SCALE
Generates 2,400+ diverse research queries with quality labels
Uses GPT-4o-mini in batches to avoid JSON parsing errors
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import List, Dict
import random
from datetime import datetime
import time
from collections import Counter

load_dotenv()

class SyntheticDataGenerator:
    """Generate synthetic research queries for training and testing"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"
        print(f"✓ Initialized with {self.model}")
        
    def generate_query_batch(self, domain: str, batch_size: int = 50) -> List[Dict]:
        """
        Generate a batch of queries (smaller batches = more reliable)
        
        Args:
            domain: Research domain
            batch_size: Number of queries in this batch
        
        Returns:
            List of query dictionaries
        """
        
        domain_descriptions = {
            'cs_ml': 'machine learning, deep learning, neural networks, AI',
            'cs_nlp': 'natural language processing, language models, text analysis',
            'cs_cv': 'computer vision, image processing, object detection',
            'biology': 'molecular biology, genetics, CRISPR, protein folding',
            'physics': 'quantum computing, quantum mechanics, particle physics',
            'medicine': 'clinical research, disease treatment, drug discovery'
        }
        
        prompt = f"""Generate {batch_size} research questions about {domain_descriptions[domain]}.

Output JSON format:
{{
  "queries": [
    {{"query": "What is deep learning", "complexity": "simple", "type": "definition"}},
    {{"query": "How do transformers work", "complexity": "moderate", "type": "methodology"}}
  ]
}}

Mix of:
- Complexity: simple, moderate, complex
- Types: definition, methodology, literature_review, comparison
- Length: 5-20 words
- Realistic academic questions

Generate {batch_size} queries:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a research query generator. Output valid JSON only.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=2000,
                temperature=0.85,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            
            # Extract queries
            if 'queries' in data:
                queries = data['queries']
            elif 'questions' in data:
                queries = data['questions']
            else:
                queries = list(data.values())[0] if data else []
            
            # Validate and enhance
            valid = []
            for q in queries:
                if isinstance(q, dict) and 'query' in q:
                    q['domain'] = domain
                    q['source'] = 'synthetic_gpt4o'
                    q.setdefault('complexity', 'moderate')
                    q.setdefault('type', 'exploratory')
                    valid.append(q)
            
            return valid
            
        except Exception as e:
            print(f"    ⚠️  Batch error: {e}")
            return []
    
    def generate_domain_queries(self, domain: str, total: int = 400) -> List[Dict]:
        """Generate all queries for a domain using batching"""
        
        batch_size = 50  # Smaller batches = more reliable
        batches_needed = (total + batch_size - 1) // batch_size
        
        all_queries = []
        
        for batch_num in range(batches_needed):
            batch_queries = self.generate_query_batch(domain, batch_size)
            all_queries.extend(batch_queries)
            
            if (batch_num + 1) % 2 == 0:
                print(f"    Progress: {len(all_queries)}/{total}")
            
            # Brief pause
            time.sleep(0.5)
        
        return all_queries[:total]  # Cap at requested amount
    
    def generate_complete_dataset(self, queries_per_domain: int = 400) -> List[Dict]:
        """Generate complete synthetic dataset"""
        
        print("="*80)
        print(f"SYNTHETIC DATA GENERATION")
        print(f"Target: {queries_per_domain * 6} total queries ({queries_per_domain} per domain)")
        print("="*80)
        
        domains = ['cs_ml', 'cs_nlp', 'cs_cv', 'biology', 'physics', 'medicine']
        all_queries = []
        
        start_time = time.time()
        
        for i, domain in enumerate(domains, 1):
            print(f"\n[{i}/6] Generating {queries_per_domain} queries for {domain}...")
            
            queries = self.generate_domain_queries(domain, queries_per_domain)
            all_queries.extend(queries)
            
            print(f"  ✓ Generated {len(queries)} queries for {domain}")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ Generation complete in {elapsed/60:.1f} minutes")
        print(f"Total queries: {len(all_queries)}")
        print(f"{'='*80}")
        
        # Statistics
        if len(all_queries) > 0:
            complexities = Counter(q.get('complexity', 'unknown') for q in all_queries)
            types = Counter(q.get('type', 'unknown') for q in all_queries)
            domains_count = Counter(q['domain'] for q in all_queries)
            
            print(f"\n📊 Dataset Statistics:")
            print(f"\nComplexity:")
            for comp, count in sorted(complexities.items()):
                pct = count/len(all_queries)*100
                print(f"  {comp:10s}: {count:4d} ({pct:5.1f}%)")
            
            print(f"\nQuery type:")
            for qtype, count in sorted(types.items()):
                pct = count/len(all_queries)*100
                print(f"  {qtype:20s}: {count:4d} ({pct:5.1f}%)")
            
            print(f"\nDomain:")
            for domain, count in sorted(domains_count.items()):
                print(f"  {domain:10s}: {count:4d}")
        
        return all_queries
    
    def add_quality_labels(self, queries: List[Dict]) -> List[Dict]:
        """Add expected quality scores"""
        
        print(f"\n🏷️  Adding quality labels to {len(queries)} queries...")
        
        for query in queries:
            # Base quality
            complexity_scores = {'simple': 0.75, 'moderate': 0.65, 'complex': 0.55}
            base = complexity_scores.get(query.get('complexity', 'moderate'), 0.65)
            
            # Type adjustment
            type_adj = {
                'definition': +0.10,
                'methodology': +0.05,
                'literature_review': -0.05,
                'comparison': 0.00
            }
            adj = type_adj.get(query.get('type', 'exploratory'), 0.0)
            
            # Noise
            noise = random.uniform(-0.08, 0.08)
            
            quality = max(0.1, min(0.95, base + adj + noise))
            
            query['expected_quality'] = round(quality, 3)
            query['expected_reward'] = round(2 * quality - 1, 3)
        
        print(f"  ✓ Labels added")
        return queries
    
    def create_train_test_split(self, queries: List[Dict], test_ratio: float = 0.2):
        """Split into train/test"""
        
        random.shuffle(queries)
        split_idx = int(len(queries) * (1 - test_ratio))
        
        train = queries[:split_idx]
        test = queries[split_idx:]
        
        print(f"\n📊 Split: Train={len(train)} | Test={len(test)}")
        
        return train, test

def main():
    """Generate complete synthetic dataset"""
    
    print("\n🚀 MAXIMUM SCALE SYNTHETIC DATA GENERATION")
    print("="*80)
    
    generator = SyntheticDataGenerator()
    
    # Generate (400 per domain = 2,400 total)
    queries = generator.generate_complete_dataset(queries_per_domain=400)
    
    if len(queries) == 0:
        print("\n❌ No queries generated - check API key")
        return None, None, None
    
    # Add labels
    queries = generator.add_quality_labels(queries)
    
    # Split
    train, test = generator.create_train_test_split(queries, test_ratio=0.2)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('data/synthetic', exist_ok=True)
    
    with open(f'data/synthetic/complete_{timestamp}.json', 'w') as f:
        json.dump(queries, f, indent=2)
    
    with open(f'data/synthetic/train_{timestamp}.json', 'w') as f:
        json.dump(train, f, indent=2)
    
    with open(f'data/synthetic/test_{timestamp}.json', 'w') as f:
        json.dump(test, f, indent=2)
    
    print(f"\n💾 Saved to data/synthetic/")
    print("\n" + "="*80)
    print("✅ COMPLETE")
    print("="*80)
    
    return queries, train, test

if __name__ == "__main__":
    queries, train, test = main()