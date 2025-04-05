#!/usr/bin/env python3
"""
Didi's Quick Search: DegenDuel Codebase Search Tool
Provides fast semantic search of the DegenDuel codebase.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_DIR, COLLECTION_NAME, DEFAULT_EMBED_MODEL

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_RESULTS = 10  # Default number of results to show

def load_index():
    """Load Didi's knowledge base."""
    # Check if database exists
    if not DB_DIR.exists():
        logger.error(f"Knowledge base directory {DB_DIR} does not exist! Run './didi.sh index' first.")
        sys.exit(1)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        logger.error(f"Collection '{COLLECTION_NAME}' not found! Run './didi.sh index' first.")
        sys.exit(1)
    
    # Set up embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=DEFAULT_EMBED_MODEL,
        max_length=512,
        trust_remote_code=True,
    )
    
    # Create vector store and index
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    return VectorStoreIndex.from_vector_store(vector_store)

def semantic_search(index, query: str, num_results: int = NUM_RESULTS):
    """Didi's semantic search on the DegenDuel codebase."""
    retriever = index.as_retriever(similarity_top_k=num_results)
    nodes = retriever.retrieve(query)
    
    if not nodes:
        print("Didi couldn't find any relevant results. 😕")
        return
    
    # Format and print results
    print(f"\nDidi found the top {len(nodes)} results for: '{query}'\n")
    print("=" * 50)
    
    for i, node in enumerate(nodes):
        score = node.score if hasattr(node, 'score') else "N/A"
        source = node.metadata.get("file_path", "Unknown source")
        
        # Get first few lines for preview
        preview = node.text.split("\n")
        preview = "\n".join(preview[:min(5, len(preview))])
        if len(node.text) > len(preview):
            preview += "\n..."
        
        print(f"\n[{i+1}] {source} (Score: {score:.4f})")
        print("-" * 50)
        print(preview)
    
    print("\n" + "=" * 50)

def main():
    """Main execution function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Didi semantic search for DegenDuel codebase')
    parser.add_argument('query', type=str, nargs='+', help='The search query')
    parser.add_argument('-n', '--num-results', type=int, default=NUM_RESULTS, 
                        help=f'Number of results to return (default: {NUM_RESULTS})')
    args = parser.parse_args()
    
    # Join query words into a single string
    query = " ".join(args.query)
    
    # Load the index
    logger.info("Loading Didi's knowledge base...")
    index = load_index()
    
    # Perform search
    semantic_search(index, query, args.num_results)

if __name__ == "__main__":
    main() 