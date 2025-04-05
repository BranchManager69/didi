#!/usr/bin/env python3
"""
Code Repository Query Script for DegenDuel
Enables semantic search and question answering over the codebase.
"""

import os
import sys
import logging
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_DIR, COLLECTION_NAME, DEFAULT_EMBED_MODEL, DEFAULT_MODEL_PATH

# LlamaIndex imports
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
TOP_K = 10  # Number of relevant chunks to retrieve

def setup_llm():
    """Set up the language model for querying."""
    logger.info(f"Loading LLM: {DEFAULT_MODEL_PATH}")
    
    # Setup model kwargs for device mapping
    model_kwargs = {
        "device_map": "auto",
    }
    
    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    
    # Create LLM instance with simpler configuration for compatibility
    llm = HuggingFaceLLM(
        model_name=DEFAULT_MODEL_PATH,
        tokenizer_name=DEFAULT_MODEL_PATH,
        context_window=4096,  # Adjust based on model
        model_kwargs=model_kwargs,
        generate_kwargs=generate_kwargs,
        system_prompt=textwrap.dedent("""
            You are an expert TypeScript and React developer working on the DegenDuel project.
            Analyze the code snippets carefully and provide detailed, accurate information.
            If you're asked about implementation details, focus on the specific code patterns and architecture.
            If you're unsure about something, mention that rather than speculating.
            Provide code examples when relevant to illustrate your points.
        """).strip(),
    )
    
    return llm

def load_index():
    """Load the existing index from ChromaDB."""
    logger.info(f"Loading index from {DB_DIR}")
    
    # Check if database exists
    if not DB_DIR.exists():
        logger.error(f"Database directory {DB_DIR} does not exist! Run the indexing script first.")
        sys.exit(1)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        logger.error(f"Collection '{COLLECTION_NAME}' not found! Run the indexing script first.")
        sys.exit(1)
    
    # Set up embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=DEFAULT_EMBED_MODEL,
        max_length=512,
        trust_remote_code=True,
    )
    
    # Create vector store and index
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

def setup_retriever(index):
    """Set up the retriever with advanced configuration."""
    # Configure the retriever for good recall
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,  # Retrieve more documents for better context
        vector_store_query_mode="default",
    )
    
    return retriever

def setup_query_engine(retriever, llm):
    """Set up the query engine with advanced features."""
    # Use a response synthesizer that can handle code well
    response_synthesizer = get_response_synthesizer(
        response_mode="refine",  # Use refine for better coherence
        llm=llm,
        verbose=True,
    )
    
    # Set up the query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # Post-process to ensure we have relevant results
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
    )
    
    return query_engine

def print_source_info(response):
    """Print information about the source documents used in the response."""
    print("\n" + "=" * 40)
    print("SOURCE DOCUMENTS:")
    print("=" * 40)
    
    if not hasattr(response, 'source_nodes') or not response.source_nodes:
        print("No source documents retrieved.")
        return
    
    for i, node in enumerate(response.source_nodes):
        score = node.score if hasattr(node, 'score') else "N/A"
        source = node.metadata.get('file_path', 'Unknown source')
        
        print(f"\n[{i+1}] {source} (Relevance: {score:.4f})")
        print("-" * 40)
        print(node.text[:300] + "..." if len(node.text) > 300 else node.text)
    
    print("\n" + "=" * 40)

def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python query_code.py 'your question about the codebase'")
        sys.exit(1)
    
    # Get query from command line arguments
    query = " ".join(sys.argv[1:])
    logger.info(f"Query: {query}")
    
    # Load the index
    index = load_index()
    
    # Set up the LLM
    llm = setup_llm()
    
    # Setup retriever
    retriever = setup_retriever(index)
    
    # Setup query engine
    query_engine = setup_query_engine(retriever, llm)
    
    # Execute query
    print("\nQuerying the codebase, please wait...\n")
    response = query_engine.query(query)
    
    # Print response
    print("\n" + "=" * 40)
    print("RESPONSE:")
    print("=" * 40)
    print(response)
    
    # Print source information
    print_source_info(response)

if __name__ == "__main__":
    main() 