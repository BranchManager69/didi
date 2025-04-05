#!/usr/bin/env python3
"""
Didi A/B Test: Embedding Model Comparison
Evaluates and compares different embedding models for code search performance.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DB_DIR, COLLECTION_NAME, EMBEDDING_MODELS, 
    METRICS_DIR, DEFAULT_EMBED_MODEL, AB_TEST_MODEL
)

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
DEFAULT_NUM_RESULTS = 10
DEFAULT_NUM_QUERIES = 5
DEFAULT_RELEVANCE_THRESHOLD = 0.7

class EmbeddingModelTester:
    """Class for testing and comparing embedding models."""
    
    def __init__(self, models_to_test: List[str] = None, collection_name: str = COLLECTION_NAME):
        """Initialize the tester with models to compare."""
        # Create metrics directory if it doesn't exist
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        self.collection_name = collection_name
        self.results = {}
        
        # Use specified models or default to comparing general vs code models
        self.models_to_test = models_to_test or [DEFAULT_EMBED_MODEL, AB_TEST_MODEL]
        logger.info(f"Models to test: {self.models_to_test}")
        
        # Initialize ChromaDB client
        if not DB_DIR.exists():
            logger.error(f"Database directory {DB_DIR} does not exist! Run indexing first.")
            sys.exit(1)
        
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
    
    def load_index(self, embed_model_name: str) -> VectorStoreIndex:
        """Load vector store index with specified embedding model."""
        logger.info(f"Loading index with embedding model: {embed_model_name}")
        start_time = time.time()
        
        # Set up the embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            max_length=512,
            trust_remote_code=True,
        )
        
        # Get collection
        try:
            collection = self.client.get_collection(name=self.collection_name)
        except ValueError:
            logger.error(f"Collection '{self.collection_name}' not found! Run indexing first.")
            sys.exit(1)
        
        # Create vector store and index
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        load_time = time.time() - start_time
        logger.info(f"Index loaded in {load_time:.2f} seconds")
        
        return index, load_time
    
    def run_query(self, index: VectorStoreIndex, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> Tuple[List, float]:
        """Run a query and measure performance."""
        start_time = time.time()
        
        # Perform search
        retriever = index.as_retriever(similarity_top_k=num_results)
        nodes = retriever.retrieve(query)
        
        query_time = time.time() - start_time
        
        return nodes, query_time
    
    def test_queries(self, queries: List[str], num_results: int = DEFAULT_NUM_RESULTS) -> Dict[str, Any]:
        """Test a set of queries against all models."""
        results = {}
        
        for model_name in self.models_to_test:
            logger.info(f"Testing model: {model_name}")
            model_results = {
                "model": model_name,
                "load_time": 0,
                "queries": [],
                "avg_query_time": 0,
                "total_query_time": 0,
            }
            
            # Load index with this model
            index, load_time = self.load_index(model_name)
            model_results["load_time"] = load_time
            
            # Run all queries
            total_query_time = 0
            for query in queries:
                logger.info(f"Running query: {query}")
                
                # Get results and timing
                nodes, query_time = self.run_query(index, query, num_results)
                total_query_time += query_time
                
                # Format results
                query_results = []
                for i, node in enumerate(nodes):
                    score = node.score if hasattr(node, 'score') else 0
                    source = node.metadata.get("file_path", "Unknown")
                    
                    query_results.append({
                        "rank": i + 1,
                        "score": score,
                        "source": source,
                        "preview": node.text[:100] + "..." if len(node.text) > 100 else node.text
                    })
                
                # Add to results
                model_results["queries"].append({
                    "query": query,
                    "time": query_time,
                    "results": query_results
                })
            
            # Calculate averages
            model_results["total_query_time"] = total_query_time
            model_results["avg_query_time"] = total_query_time / len(queries) if queries else 0
            
            results[model_name] = model_results
        
        return results
    
    def compare_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results between models."""
        if len(results) < 2:
            logger.warning("Need at least 2 models to compare")
            return {}
        
        comparison = {
            "models_compared": list(results.keys()),
            "query_count": len(next(iter(results.values()))["queries"]),
            "timestamp": datetime.now().isoformat(),
            "performance": {},
            "result_overlap": {},
            "result_difference": {},
            "queries": {}
        }
        
        # Compare performance metrics
        for model_name, model_results in results.items():
            comparison["performance"][model_name] = {
                "load_time": model_results["load_time"],
                "total_query_time": model_results["total_query_time"],
                "avg_query_time": model_results["avg_query_time"]
            }
        
        # Compare results for each query
        model_names = list(results.keys())
        for i, query_data in enumerate(results[model_names[0]]["queries"]):
            query = query_data["query"]
            comparison["queries"][query] = {}
            
            # Get top results from each model
            model_sources = {}
            for model_name in model_names:
                model_query_results = results[model_name]["queries"][i]["results"]
                model_sources[model_name] = [r["source"] for r in model_query_results]
            
            # Calculate overlap and differences
            for j, model1 in enumerate(model_names):
                for model2 in model_names[j+1:]:
                    overlap_key = f"{model1}_vs_{model2}"
                    
                    # Get top sources from each model
                    sources1 = set(model_sources[model1])
                    sources2 = set(model_sources[model2])
                    
                    # Calculate overlap
                    overlap = sources1.intersection(sources2)
                    only_in_1 = sources1 - sources2
                    only_in_2 = sources2 - sources1
                    
                    # Record overlap statistics
                    if overlap_key not in comparison["result_overlap"]:
                        comparison["result_overlap"][overlap_key] = []
                        comparison["result_difference"][overlap_key] = []
                    
                    overlap_pct = len(overlap) / len(sources1.union(sources2)) if sources1 or sources2 else 0
                    
                    comparison["result_overlap"][overlap_key].append(overlap_pct)
                    comparison["result_difference"][overlap_key].append({
                        "query": query,
                        "overlap_count": len(overlap),
                        "overlap_percent": overlap_pct * 100,
                        f"only_in_{model1}": list(only_in_1),
                        f"only_in_{model2}": list(only_in_2)
                    })
                    
                    # Record query-specific data
                    comparison["queries"][query][overlap_key] = {
                        "overlap_percent": overlap_pct * 100,
                        "common_results": list(overlap),
                        f"unique_to_{model1}": list(only_in_1),
                        f"unique_to_{model2}": list(only_in_2)
                    }
        
        # Calculate average overlap across all queries
        for key, overlaps in comparison["result_overlap"].items():
            comparison["result_overlap"][key] = sum(overlaps) / len(overlaps) if overlaps else 0
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], comparison: Dict[str, Any]):
        """Save test results and comparison to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual model results
        results_file = METRICS_DIR / f"embed_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save comparison
        comparison_file = METRICS_DIR / f"embed_test_comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Comparison saved to {comparison_file}")
        
        # Also save a summary text file
        summary_file = METRICS_DIR / f"embed_test_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Didi A/B Testing Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Models Compared:\n")
            for model in comparison["models_compared"]:
                f.write(f"- {model}\n")
            f.write("\n")
            
            f.write("Performance Metrics:\n")
            f.write("-"*40 + "\n")
            for model, perf in comparison["performance"].items():
                f.write(f"{model}:\n")
                f.write(f"  Load time: {perf['load_time']:.2f} seconds\n")
                f.write(f"  Avg query time: {perf['avg_query_time']:.4f} seconds\n")
            f.write("\n")
            
            f.write("Result Overlap:\n")
            f.write("-"*40 + "\n")
            for key, overlap in comparison["result_overlap"].items():
                model1, model2 = key.split("_vs_")
                f.write(f"{model1} vs {model2}: {overlap*100:.1f}% average overlap\n")
            f.write("\n")
            
            f.write("Queries:\n")
            f.write("-"*40 + "\n")
            for query, query_data in comparison["queries"].items():
                f.write(f"Query: \"{query}\"\n")
                for key, data in query_data.items():
                    model1, model2 = key.split("_vs_")
                    f.write(f"  {model1} vs {model2}: {data['overlap_percent']:.1f}% overlap\n")
                    f.write(f"  Unique to {model1}: {len(data[f'unique_to_{model1}'])}\n")
                    f.write(f"  Unique to {model2}: {len(data[f'unique_to_{model2}'])}\n")
                f.write("\n")
        
        logger.info(f"Summary saved to {summary_file}")
        return results_file, comparison_file, summary_file
    
    def run_test(self, queries: List[str], num_results: int = DEFAULT_NUM_RESULTS):
        """Run the complete test and save results."""
        # Run tests
        results = self.test_queries(queries, num_results)
        
        # Compare results
        comparison = self.compare_results(results)
        
        # Save results
        return self.save_results(results, comparison)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='A/B test embedding models for Didi')
    parser.add_argument('--queries', type=str, nargs='+', help='Queries to test')
    parser.add_argument('--query-file', type=str, help='File containing queries to test (one per line)')
    parser.add_argument('--num-results', type=int, default=DEFAULT_NUM_RESULTS, 
                        help=f'Number of results to retrieve (default: {DEFAULT_NUM_RESULTS})')
    parser.add_argument('--models', type=str, nargs='+', help='Models to test (model names or paths)')
    args = parser.parse_args()
    
    # Get queries
    queries = []
    if args.query_file:
        with open(args.query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    elif args.queries:
        queries = args.queries
    else:
        # Example queries if none provided
        queries = [
            "websocket connection",
            "user authentication",
            "contest system",
            "wallet integration",
            "loading state handling",
        ]
    
    if len(queries) == 0:
        logger.error("No queries provided!")
        sys.exit(1)
    
    # Get models
    models = args.models if args.models else None  # Use default models if none specified
    
    # Initialize tester
    tester = EmbeddingModelTester(models)
    
    # Run test
    results_file, comparison_file, summary_file = tester.run_test(queries, args.num_results)
    
    # Print summary path
    print(f"\nTest complete! Summary available at: {summary_file}\n")

if __name__ == "__main__":
    main() 