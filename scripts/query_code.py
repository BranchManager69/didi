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
from config import CONTEXT_WINDOW, MAX_NEW_TOKENS, TEMPERATURE, get_current_profile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration is now imported from config.py
# Use the values from the current profile
TOP_K = 10  # Number of relevant chunks to retrieve

# Get current profile info for logging
current_profile = get_current_profile()
logger.info(f"Using profile: {current_profile['profile_name']}")
logger.info(f"LLM model: {DEFAULT_MODEL_PATH}")
logger.info(f"Embedding model: {DEFAULT_EMBED_MODEL}")
logger.info(f"Collection name: {COLLECTION_NAME}")
logger.info(f"Context window: {CONTEXT_WINDOW}")
logger.info(f"Max new tokens: {MAX_NEW_TOKENS}")
logger.info(f"Temperature: {TEMPERATURE}")

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
    """Check if all required dependencies are installed and GPU is properly configured."""
    try:
        import importlib.util
        
        required_modules = [
            "llama_index",
            "chromadb",
            "sentence_transformers",
            "torch",
            "transformers"
        ]
        
        # Check if Ollama is required
        use_ollama = os.environ.get("DIDI_USE_OLLAMA", "").lower() == "true"
        if use_ollama:
            required_modules.append("ollama")
        
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
            
            if "ollama" in missing:
                print_colored("\nFor Ollama integration:", "yellow")
                print_colored("pip install ollama", "yellow")
                print_colored("bash setup_ollama.sh  # To install and set up Ollama", "yellow")
            
            print_colored("\nOr run the following command to install all dependencies:", "yellow")
            print_colored("pip install -r requirements.txt", "yellow")
            return False
        
        # Check GPU availability for PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                mem_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
                
                print_colored("\nGPU detected! 🎮", "green")
                print_colored(f"Device: {device_name}", "green")
                print_colored(f"Memory: {mem_total:.1f} GB", "green")
                print_colored(f"CUDA version: {torch.version.cuda}", "green")
                
                # Check if the current profile's model is suitable for this GPU
                if "gh200" in current_profile['profile_name'] and mem_total < 400:
                    print_colored("\nWarning: You're using the gh200 profile but your GPU has less than 400GB of memory.", "yellow")
                    print_colored("This profile is designed for the NVIDIA GH200 with 480GB memory.", "yellow")
                    print_colored("Consider switching to a smaller model if you encounter out-of-memory errors.", "yellow")
                elif "powerful" in current_profile['profile_name'] and mem_total < 40:
                    print_colored("\nWarning: You're using the powerful profile but your GPU has less than 40GB of memory.", "yellow")
                    print_colored("Consider switching to the 'default' or 'balanced' profile to avoid out-of-memory errors.", "yellow")
            else:
                print_colored("\nNo GPU detected. Running in CPU mode (will be much slower).", "yellow")
                print_colored("If you have a GPU, check your CUDA environment variables and PyTorch installation.", "yellow")
                
                # Check for large models on CPU
                if any(x in current_profile['profile_name'] for x in ['gh200', 'powerful']):
                    print_colored("\nWarning: You're trying to run a large model profile without a GPU.", "red")
                    print_colored("This will likely fail or be extremely slow. Please switch to the 'default' profile.", "red")
                    print_colored("Run: ./didi.sh profile switch default", "red")
        except Exception as e:
            print_colored(f"\nWarning: Could not check GPU status: {e}", "yellow")
            
        # All dependencies are available
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
    """Set up the language model for Didi based on the current profile."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core import Settings
    
    # Prepare system prompt for code understanding
    system_prompt = textwrap.dedent("""
        You are Didi, the friendly and knowledgeable AI assistant for DegenDuel.
        You specialize in the DegenDuel codebase, which is a TypeScript and React-based web application.
        
        When answering questions:
        - Be concise, friendly, and helpful
        - Use emojis occasionally to add personality 
        - If you're unsure about something, be honest rather than speculating
        - Reference specific code files when explaining implementation details
        - Provide code examples when relevant to illustrate your points
        
        Remember that your purpose is to help DegenDuel developers understand and navigate their codebase!
    """).strip()
    
    # Check if we should use Ollama
    use_ollama = os.environ.get("DIDI_USE_OLLAMA", "").lower() == "true"
    
    if use_ollama:
        # Use Ollama with the specified model
        try:
            from llm_ollama import OllamaLLM
            
            # Get Ollama model and URL from environment variables or use defaults
            ollama_model = os.environ.get("OLLAMA_MODEL", "llama4")
            ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
            
            logger.info(f"Loading Didi's brain using Ollama with model: {ollama_model}")
            
            # Use profile settings for Ollama configuration
            llm = OllamaLLM(
                model_name=ollama_model,
                ollama_url=ollama_url,
                temperature=TEMPERATURE,  # From profile
                max_tokens=MAX_NEW_TOKENS,  # From profile
                context_window=CONTEXT_WINDOW,  # From profile
                system_prompt=system_prompt,
            )
            
            logger.info(f"Using Ollama LLM with {ollama_model} model")
            logger.info(f"Using context window: {CONTEXT_WINDOW}")
            logger.info(f"Using max tokens: {MAX_NEW_TOKENS}")
            logger.info(f"Using temperature: {TEMPERATURE}")
            
        except ImportError:
            logger.error("Failed to import OllamaLLM. Make sure ollama is installed (pip install ollama)")
            logger.error("Falling back to HuggingFace model")
            use_ollama = False
    
    if not use_ollama:
        # Use HuggingFace model
        from llama_index.llms.huggingface import HuggingFaceLLM
        
        logger.info(f"Loading Didi's brain: {DEFAULT_MODEL_PATH}")
        
        # Setup model kwargs for optimal GPU usage
        model_kwargs = {
            "torch_dtype": "auto",  # Use the best precision for the device
        }
        
        # Configure generation parameters from profile settings
        generate_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,  # From profile
            "temperature": TEMPERATURE,  # From profile
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        
        # Create LLM instance with configuration from profile
        llm = HuggingFaceLLM(
            model_name=DEFAULT_MODEL_PATH,
            tokenizer_name=DEFAULT_MODEL_PATH,
            context_window=CONTEXT_WINDOW,  # From profile
            model_kwargs=model_kwargs,
            generate_kwargs=generate_kwargs,
            system_prompt=system_prompt,
        )
    
    # Configure global settings to use our LLM instead of OpenAI
    # This prevents fallback to OpenAI in various llama-index components
    Settings.llm = llm
    logger.info("Configured global Settings to use LLM")
    
    return llm

def load_index():
    """Load Didi's knowledge base from ChromaDB using the current profile settings."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core import Settings
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    
    # Use the global DB_DIR variable and allow modification
    global DB_DIR
    
    logger.info(f"Loading Didi's knowledge base from {DB_DIR}")
    logger.info(f"Using collection: {COLLECTION_NAME}")
    
    # Check if database exists
    if not DB_DIR.exists():
        alternate_db_dir = Path("/home/ubuntu/degenduel-gpu/data/chroma_db")
        if alternate_db_dir.exists():
            # Use alternate path if default doesn't exist
            logger.info(f"Using alternate knowledge base directory: {alternate_db_dir}")
            DB_DIR = alternate_db_dir
        else:
            logger.error(f"Knowledge base directory {DB_DIR} does not exist! Run './didi.sh index' first.")
            sys.exit(1)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Found collection '{COLLECTION_NAME}' with {collection.count()} documents")
    except ValueError:
        logger.error(f"Collection '{COLLECTION_NAME}' not found! Run './didi.sh index' first.")
        logger.error(f"Make sure you've indexed the code with the same profile ('{current_profile['profile_name']}').")
        logger.error(f"Available collections: {[c.name for c in client.list_collections()]}")
        sys.exit(1)
    
    # Set up embedding model from profile
    max_length = 512  # Default for most embedding models
    if "bge" in DEFAULT_EMBED_MODEL.lower():
        max_length = 1024  # BGE models support longer sequences
    
    logger.info(f"Setting up embedding model: {DEFAULT_EMBED_MODEL}")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=DEFAULT_EMBED_MODEL,
        max_length=max_length,
        trust_remote_code=True,
    )
    
    # Create vector store and index
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    logger.info("Vector store initialized successfully")
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

def setup_retriever(index):
    """Set up the retriever with advanced configuration based on profile settings."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core.retrievers import VectorIndexRetriever
    
    # Import CHUNK_SIZE from config if available
    # Use safe defaults if not available
    chunk_size = 1024
    try:
        from config import CHUNK_SIZE
        chunk_size = CHUNK_SIZE
    except ImportError:
        logger.warning("Could not import CHUNK_SIZE from config, using default value")
    
    # Adjust TOP_K based on the context window and chunk size to optimize context usage
    # For larger context windows and chunk sizes, we can retrieve fewer documents
    retrieval_k = TOP_K
    if CONTEXT_WINDOW >= 8192 and chunk_size >= 4096:
        # For large context windows and chunks, retrieve fewer but larger chunks
        retrieval_k = 6
    elif CONTEXT_WINDOW >= 4096 and chunk_size >= 2048:
        # For medium context windows and chunks, retrieve a balanced number
        retrieval_k = 8
    
    logger.info(f"Configuring retriever to fetch top {retrieval_k} chunks")
    logger.info(f"Using chunk size: {chunk_size}")
    
    # Configure the retriever for good recall
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=retrieval_k,
        vector_store_query_mode="default",
    )
    
    return retriever

def setup_query_engine(retriever, llm):
    """Set up Didi's query engine with advanced features based on profile settings."""
    # Import modules here to avoid import errors during dependency check
    from llama_index.core import get_response_synthesizer
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    
    # Use a response synthesizer that can handle code well
    # Explicitly use TreeSummarize to avoid OpenAI dependency
    from llama_index.core.response_synthesizers import TreeSummarize
    
    # Adjust similarity cutoff based on the embedding model
    # BGE models tend to give higher similarity scores
    similarity_cutoff = 0.7
    if "bge" in DEFAULT_EMBED_MODEL.lower():
        similarity_cutoff = 0.75
    elif "e5" in DEFAULT_EMBED_MODEL.lower():
        similarity_cutoff = 0.65
    elif "code" in DEFAULT_EMBED_MODEL.lower():
        similarity_cutoff = 0.6  # Code models are more specific
    
    logger.info(f"Using similarity cutoff: {similarity_cutoff}")
    
    # Create response synthesizer with explicit mode to avoid OpenAI
    # Note: Different versions of llama_index may have different parameters
    # for TreeSummarize, so we use a try/except block to handle both cases
    try:
        # Try the newer API (if available)
        chunk_size_limit = min(CONTEXT_WINDOW // 2, 3000)
        response_synthesizer = TreeSummarize(
            llm=llm,
            verbose=True,
            chunk_size_limit=chunk_size_limit
        )
        logger.info(f"Using TreeSummarize response synthesizer with chunk size limit: {chunk_size_limit}")
    except TypeError:
        # Fall back to the older API
        response_synthesizer = TreeSummarize(
            llm=llm,
            verbose=True
        )
        logger.info("Using TreeSummarize response synthesizer (without chunk size limit)")
    
    # Set up the query engine with profile-optimized settings
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # Post-process to ensure we have relevant results
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
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
    
    # Check if we're using Ollama
    use_ollama = os.environ.get("DIDI_USE_OLLAMA", "").lower() == "true"
    if use_ollama:
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama4")
        logger.info(f"Using Ollama with {ollama_model} model")
    
    logger.info(f"Question for Didi: {query}")
    
    # Display GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            mem_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
            logger.info(f"GPU available: {device_name} (Memory: {mem_total:.1f} GB)")
        else:
            logger.info("No GPU available. Using CPU mode.")
    except Exception as e:
        logger.warning(f"Could not determine GPU status: {e}")
    
    # Check dependencies first
    if not check_dependencies():
        mock_answer(query)
        return
    
    try:
        # Load the index
        index = load_index()
        
        # Set up the LLM (also configures global Settings)
        llm = setup_llm()
        
        # Setup retriever
        retriever = setup_retriever(index)
        
        # Setup query engine
        query_engine = setup_query_engine(retriever, llm)
        
        # Execute query
        if use_ollama:
            ollama_model = os.environ.get("OLLAMA_MODEL", "llama4")
            print_colored(f"\nDidi with {ollama_model.upper()} is thinking about your question, please wait... 🤔\n", "yellow")
        else:
            # Include profile info in the thinking message
            model_name = current_profile.get("llm_model", DEFAULT_MODEL_PATH).split("/")[-1]
            print_colored(f"\nDidi with {model_name} (profile: {current_profile['profile_name']}) is thinking about your question, please wait... 🤔\n", "yellow")
        
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