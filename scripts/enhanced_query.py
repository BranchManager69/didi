#!/usr/bin/env python3
"""
Didi's Brain: DegenDuel's AI Assistant
Enables semantic search and question answering over the DegenDuel codebase.
Enhanced version with improved performance and features.
"""

import os
import sys
import logging
import textwrap
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_DIR, COLLECTION_NAME, DEFAULT_EMBED_MODEL, DEFAULT_MODEL_PATH, MODEL_CACHE_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
TOP_K = 15  # Increased number of relevant chunks to retrieve
RERANK_TOP_N = 10  # Number of chunks to keep after reranking

# ANSI colors for output
COLORS = {
    'green': '\033[0;32m',
    'blue': '\033[0;34m',
    'yellow': '\033[1;33m',
    'red': '\033[0;31m',
    'purple': '\033[0;35m',
    'cyan': '\033[0;36m',
    'bold': '\033[1m',
    'reset': '\033[0m'
}

def print_colored(text, color=None):
    """Print text with ANSI color codes."""
    if color and color in COLORS:
        print(f"{COLORS[color]}{text}{COLORS['reset']}")
    else:
        print(text)

def print_header(text, color='blue'):
    """Print a formatted header."""
    print()
    print_colored("=" * 60, color)
    print_colored(f" {text} ", color)
    print_colored("=" * 60, color)
    print()

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import importlib.util
        
        required_modules = [
            "llama_index",
            "chromadb",
            "sentence_transformers",
            "torch",
            "transformers"
        ]
        
        missing = []
        for module in required_modules:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        
        if missing:
            print_header("MISSING DEPENDENCIES", "red")
            for module in missing:
                print_colored(f"  - {module}", "red")
            print_colored("\nPlease install the missing dependencies with:", "yellow")
            print_colored(f"pip install {' '.join(missing)}", "yellow")
            print_colored("\nOr run the following command to install all dependencies:", "yellow")
            print_colored("pip install -r requirements.txt", "yellow")
            return False
            
        return True
    except Exception as e:
        print_colored(f"Error checking dependencies: {e}", "red")
        return False

def setup_llm():
    """Set up the language model for Didi with optimization for Lambda Labs."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.llms.huggingface import HuggingFaceLLM
    
    logger.info(f"Loading Didi's brain: {DEFAULT_MODEL_PATH}")
    
    # Setup model kwargs for device mapping
    # Optimize for Lambda Labs GPU instances
    model_kwargs = {
        "device_map": "auto",
        "load_in_8bit": True,  # More efficient memory usage
    }
    
    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    
    # Create LLM instance with system prompt focused on code understanding
    llm = HuggingFaceLLM(
        model_name=DEFAULT_MODEL_PATH,
        tokenizer_name=DEFAULT_MODEL_PATH,
        context_window=4096,  # Adjust based on model
        model_kwargs=model_kwargs,
        generate_kwargs=generate_kwargs,
        cache_dir=str(MODEL_CACHE_DIR),  # Use cached models
        system_prompt=textwrap.dedent("""
            You are Didi, the friendly and knowledgeable AI assistant for DegenDuel.
            You specialize in the DegenDuel codebase, which is a TypeScript and React-based web application for a crypto trading game platform.
            
            When answering questions:
            - Be concise, clear, and accurate
            - Provide technical details that would help a developer understand the implementation
            - Reference specific code files and line numbers when explaining implementation details
            - Relate components to each other to explain the overall architecture
            - If you're unsure about something, be honest rather than speculating
            - Show code snippets when they help illustrate your explanations
            
            Remember that your purpose is to help DegenDuel developers understand, navigate, and improve their codebase!
        """).strip(),
    )
    
    return llm

def load_index():
    """Load Didi's knowledge base from ChromaDB with optimizations."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core import Settings
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    
    logger.info(f"Loading Didi's knowledge base from {DB_DIR}")
    start_time = time.time()
    
    # Check if database exists
    if not DB_DIR.exists():
        logger.error(f"Knowledge base directory {DB_DIR} does not exist! Run indexing first.")
        sys.exit(1)
    
    # Initialize ChromaDB with optimizations
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        logger.error(f"Collection '{COLLECTION_NAME}' not found! Run indexing first.")
        sys.exit(1)
    
    # Set up embedding model with caching
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=DEFAULT_EMBED_MODEL,
        max_length=512,
        trust_remote_code=True,
        cache_folder=str(MODEL_CACHE_DIR),  # Use cached models
    )
    
    # Create vector store and index
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    
    load_time = time.time() - start_time
    logger.info(f"Knowledge base loaded in {load_time:.2f} seconds")
    
    return index

def setup_hybrid_retriever(index):
    """Set up a hybrid retriever that combines vector search with keyword search."""
    try:
        # Import modules for hybrid retrieval
        from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever
        
        # Vector retriever (semantic search)
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=TOP_K,
            vector_store_query_mode="default", 
        )
        
        # BM25 retriever (keyword search)
        try:
            bm25_retriever = BM25Retriever.from_vector_index(
                index,
                similarity_top_k=TOP_K,
            )
            
            # Combine both retrievers with reciprocal rank fusion
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=RERANK_TOP_N,  # Keep top N after fusion
                fusion_type="reciprocal_rank",
            )
            logger.info("Using hybrid retrieval (vector + BM25)")
            
        except Exception as e:
            logger.warning(f"Could not initialize BM25 retriever: {e}")
            logger.info("Falling back to vector retrieval only")
            retriever = vector_retriever
            
    except ImportError:
        # Fall back to vector retriever if hybrid retrieval is not available
        from llama_index.core.retrievers import VectorIndexRetriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=TOP_K,
        )
        logger.info("Using vector retrieval only")
    
    return retriever

def setup_query_engine(retriever, llm):
    """Set up Didi's query engine with advanced features and optimized for code Q&A."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core import get_response_synthesizer
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor

    # Enhanced response synthesizer for code understanding
    try:
        # Try to use advanced response synthesizer
        from llama_index.core.response_synthesizers import TreeSummarize
        response_synthesizer = TreeSummarize(
            llm=llm,
            verbose=True,
        )
        logger.info("Using TreeSummarize response synthesizer")
    except ImportError:
        # Fall back to standard response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",  # More concise responses
            llm=llm,
            verbose=True,
        )
        logger.info("Using standard response synthesizer")
    
    # Set up the query engine with optimization for code questions
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # Post-process to ensure we have relevant results
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.65)  # Lower threshold to include more potentially relevant results
        ],
    )
    
    return query_engine

def print_source_info(response):
    """Print information about the source documents used in the response."""
    print_header("DIDI'S SOURCES", "green")
    
    if not hasattr(response, 'source_nodes') or not response.source_nodes:
        print_colored("No source documents retrieved.", "yellow")
        return
    
    for i, node in enumerate(response.source_nodes):
        score = node.score if hasattr(node, 'score') else "N/A"
        source = node.metadata.get('file_path', 'Unknown source')
        repo_name = node.metadata.get('repo_name', 'Unknown repository')
        
        print_colored(f"[{i+1}] {source}", "cyan")
        print_colored(f"    Repository: {repo_name}", "purple")
        print_colored(f"    Relevance: {score:.4f}", "yellow")
        print_colored("-" * 60, "blue")
        
        # Format code snippets with a max width of 80 characters
        text = node.text
        if len(text) > 400:
            text = text[:400] + "..."
        print(text)
        print()
    
    print_colored("-" * 60, "blue")

def interactive_mode():
    """Run Didi in interactive mode to answer multiple questions."""
    # Check dependencies first
    if not check_dependencies():
        print_colored("Cannot start interactive mode due to missing dependencies.", "red")
        return
    
    print_header("DIDI INTERACTIVE MODE", "cyan")
    print_colored("Welcome to Didi, DegenDuel's AI Assistant!", "yellow")
    print_colored("I can answer questions about the DegenDuel codebase.", "yellow")
    print_colored("Type 'exit', 'quit', or 'q' to exit.", "yellow")
    print()
    
    try:
        # Load the index
        index = load_index()
        
        # Set up the LLM
        llm = setup_llm()
        
        # Setup retriever
        retriever = setup_hybrid_retriever(index)
        
        # Setup query engine
        query_engine = setup_query_engine(retriever, llm)
        
        while True:
            print_colored("\nAsk a question about the DegenDuel codebase:", "bold")
            query = input("> ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                print_colored("\nThank you for using Didi! Goodbye!", "yellow")
                break
            
            if not query.strip():
                continue
                
            try:
                # Execute query
                print_colored("\nDidi is thinking about your question, please wait...", "yellow")
                start_time = time.time()
                response = query_engine.query(query)
                query_time = time.time() - start_time
                
                # Print response
                print_header("DIDI'S ANSWER", "blue")
                print(response)
                print_colored(f"\nResponse generated in {query_time:.2f} seconds", "green")
                
                # Print source information
                print_source_info(response)
                
            except Exception as e:
                print_colored(f"\nError answering question: {e}", "red")
    
    except ImportError as e:
        print_colored(f"\nError loading dependencies: {e}", "red")
        print_colored("Didi is missing some required packages.", "yellow")
    except Exception as e:
        print_colored(f"\nError in interactive mode: {e}", "red")

def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-i', '--interactive']:
            interactive_mode()
            return
        else:
            # Get query from command line arguments
            query = " ".join(sys.argv[1:])
    else:
        print_colored("Usage:", "yellow")
        print_colored("  python enhanced_query.py 'your question about the DegenDuel codebase'", "yellow")
        print_colored("  python enhanced_query.py -i  # Interactive mode", "yellow")
        return
    
    logger.info(f"Question for Didi: {query}")
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    try:
        # Load the index
        index = load_index()
        
        # Set up the LLM
        llm = setup_llm()
        
        # Setup retriever
        retriever = setup_hybrid_retriever(index)
        
        # Setup query engine
        query_engine = setup_query_engine(retriever, llm)
        
        # Execute query
        print_colored("\nDidi is thinking about your question, please wait...", "yellow")
        start_time = time.time()
        response = query_engine.query(query)
        query_time = time.time() - start_time
        
        # Print response
        print_header("DIDI'S ANSWER", "blue")
        print(response)
        print_colored(f"\nResponse generated in {query_time:.2f} seconds", "green")
        
        # Print source information
        print_source_info(response)
    
    except ImportError as e:
        print_colored(f"\nError loading dependencies: {e}", "red")
        print_colored("Didi is missing some required packages.", "yellow")
    except Exception as e:
        print_colored(f"\nError: {e}", "red")
        print_colored("Didi encountered an error while answering your question.", "yellow")

if __name__ == "__main__":
    main()