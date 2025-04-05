#!/usr/bin/env python3
"""
Didi's Brain: DegenDuel's AI Assistant
Enables semantic search and question answering over the DegenDuel codebase.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
TOP_K = 10  # Number of relevant chunks to retrieve

# ANSI colors for output
COLORS = {
    'green': '\033[0;32m',
    'blue': '\033[0;34m',
    'yellow': '\033[1;33m',
    'red': '\033[0;31m',
    'reset': '\033[0m'
}

def print_colored(text, color=None):
    """Print text with ANSI color codes."""
    if color and color in COLORS:
        print(f"{COLORS[color]}{text}{COLORS['reset']}")
    else:
        print(text)

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
            print_colored("Missing required dependencies:", "red")
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

def mock_answer(query: str):
    """Provide a mock answer when dependencies are missing."""
    print_colored("\n" + "=" * 40, "blue")
    print_colored("DIDI'S ANSWER:", "yellow")
    print_colored("=" * 40, "blue")
    print_colored(f"\nTo answer your question about \"{query}\", I need my full knowledge base and language models installed.", "yellow")
    print_colored("\nCurrently, some of my required dependencies are missing. These are specialized libraries that help me understand code and answer questions intelligently.", "yellow")
    
    print_colored("\nHere's what you could find if the dependencies were installed:", "blue")
    print_colored("\n1. Relevant code files related to your question", "green")
    print_colored("2. How the code is structured and implemented", "green")
    print_colored("3. Explanations about how specific features work", "green")
    print_colored("4. Examples and best practices from the codebase", "green")
    
    print_colored("\nPlease install the missing dependencies to unlock my full capabilities! 🤓", "yellow")
    
    # Mock source information
    print_colored("\n" + "=" * 40, "blue")
    print_colored("EXAMPLE SOURCES I MIGHT USE:", "yellow")
    print_colored("=" * 40, "blue")
    
    print_colored("\n[1] src/hooks/useWebSocket.ts (Relevance: 0.92)", "green")
    print_colored("-" * 40, "blue")
    print_colored("// This would show real code related to your query...")
    
    print_colored("\n[2] src/contexts/WebSocketContext.tsx (Relevance: 0.86)", "green")
    print_colored("-" * 40, "blue")
    print_colored("// This would show another relevant code snippet...")
    
    print_colored("\n" + "=" * 40, "blue")

def setup_llm():
    """Set up the language model for Didi."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.llms.huggingface import HuggingFaceLLM
    
    logger.info(f"Loading Didi's brain: {DEFAULT_MODEL_PATH}")
    
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
            You are Didi, the friendly and knowledgeable AI assistant for DegenDuel.
            You specialize in the DegenDuel codebase, which is a TypeScript and React-based web application.
            
            When answering questions:
            - Be concise, friendly, and helpful
            - Use emojis occasionally to add personality 
            - If you're unsure about something, be honest rather than speculating
            - Reference specific code files when explaining implementation details
            - Provide code examples when relevant to illustrate your points
            
            Remember that your purpose is to help DegenDuel developers understand and navigate their codebase!
        """).strip(),
    )
    
    return llm

def load_index():
    """Load Didi's knowledge base from ChromaDB."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core import Settings
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    
    logger.info(f"Loading Didi's knowledge base from {DB_DIR}")
    
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
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

def setup_retriever(index):
    """Set up the retriever with advanced configuration."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core.retrievers import VectorIndexRetriever
    
    # Configure the retriever for good recall
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,  # Retrieve more documents for better context
        vector_store_query_mode="default",
    )
    
    return retriever

def setup_query_engine(retriever, llm):
    """Set up Didi's query engine with advanced features."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core import get_response_synthesizer
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    
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
    print_colored("\n" + "=" * 40, "blue")
    print_colored("DIDI'S SOURCES:", "yellow")
    print_colored("=" * 40, "blue")
    
    if not hasattr(response, 'source_nodes') or not response.source_nodes:
        print_colored("No source documents retrieved.", "yellow")
        return
    
    for i, node in enumerate(response.source_nodes):
        score = node.score if hasattr(node, 'score') else "N/A"
        source = node.metadata.get('file_path', 'Unknown source')
        
        print_colored(f"\n[{i+1}] {source} (Relevance: {score:.4f})", "green")
        print_colored("-" * 40, "blue")
        print(node.text[:300] + "..." if len(node.text) > 300 else node.text)
    
    print_colored("\n" + "=" * 40, "blue")

def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print_colored("Usage: python query_code.py 'your question about the DegenDuel codebase'", "yellow")
        sys.exit(1)
    
    # Get query from command line arguments
    query = " ".join(sys.argv[1:])
    logger.info(f"Question for Didi: {query}")
    
    # Check dependencies first
    if not check_dependencies():
        mock_answer(query)
        return
    
    try:
        # Load the index
        index = load_index()
        
        # Set up the LLM
        llm = setup_llm()
        
        # Setup retriever
        retriever = setup_retriever(index)
        
        # Setup query engine
        query_engine = setup_query_engine(retriever, llm)
        
        # Execute query
        print_colored("\nDidi is thinking about your question, please wait... 🤔\n", "yellow")
        response = query_engine.query(query)
        
        # Print response
        print_colored("\n" + "=" * 40, "blue")
        print_colored("DIDI'S ANSWER:", "yellow")
        print_colored("=" * 40, "blue")
        print(response)
        
        # Print source information
        print_source_info(response)
    
    except ImportError as e:
        print_colored(f"\nError loading dependencies: {e}", "red")
        print_colored("Didi is missing some required packages.", "yellow")
        mock_answer(query)
    except Exception as e:
        print_colored(f"\nError: {e}", "red")
        print_colored("Didi encountered an error while answering your question.", "yellow")
        mock_answer(query)

if __name__ == "__main__":
    main()