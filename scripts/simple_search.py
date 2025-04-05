#!/usr/bin/env python3
"""
Didi's Detailed Search: DegenDuel Codebase Explorer
Provides comprehensive code snippets with full context for deeper understanding.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_DIR, COLLECTION_NAME, DEFAULT_EMBED_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_RESULTS = 15  # Default number of results to show

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
            "sentence_transformers"
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

def mock_detailed_search(query: str, num_results: int = NUM_RESULTS):
    """Fallback mock detailed search function when dependencies are missing."""
    print_colored(f"\nDidi would provide detailed code for: '{query}' if the search dependencies were installed.\n", "yellow")
    print_colored("Example detailed results would look like:\n", "blue")
    
    print_colored("="*80, "blue")
    
    print_colored("\n[1] src/contexts/WebSocketContext.tsx (Score: 0.89)", "green")
    print_colored("-"*80, "blue")
    print_colored("import React, { createContext, useContext, useState, useEffect } from 'react';\n\n")
    print_colored("interface WebSocketContextType {\n  socket: WebSocket | null;\n  connected: boolean;\n  connect: () => void;\n  disconnect: () => void;\n  send: (message: any) => void;\n}\n\n")
    print_colored("const WebSocketContext = createContext<WebSocketContextType | null>(null);\n\n")
    print_colored("export const WebSocketProvider: React.FC = ({ children }) => {\n  // Full implementation would be shown here\n  ...\n}\n\n")
    
    print_colored("\n[2] src/hooks/useWebSocket.ts (Score: 0.84)", "green")
    print_colored("-"*80, "blue")
    print_colored("import { useState, useEffect, useCallback } from 'react';\n\n")
    print_colored("export const useWebSocket = (url: string) => {\n  // Full implementation would be shown here\n  ...\n}\n\n")
    
    print_colored("\nTo see actual detailed results, please install the required dependencies.", "yellow")

def load_index():
    """Load Didi's knowledge base."""
    # Import modules here to allow dependency check to work
    from llama_index.core import Settings
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    
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

def detailed_search(index, query: str, num_results: int = NUM_RESULTS):
    """Didi's detailed search with full code context."""
    retriever = index.as_retriever(similarity_top_k=num_results)
    nodes = retriever.retrieve(query)
    
    if not nodes:
        print_colored("Didi couldn't find any relevant results. 😕", "yellow")
        return
    
    # Format and print results
    print_colored(f"\nDidi found the top {len(nodes)} detailed results for: '{query}'\n", "green")
    print_colored("="*80, "blue")
    
    for i, node in enumerate(nodes):
        score = node.score if hasattr(node, 'score') else "N/A"
        source = node.metadata.get("file_path", "Unknown source")
        
        print_colored(f"\n[{i+1}] {source} (Score: {score:.4f})", "green")
        print_colored("-"*80, "blue")
        
        # Print the entire content for detailed analysis
        print(node.text)
        print_colored("-"*80, "blue")
    
    print_colored("\n" + "="*80, "blue")

def main():
    """Main execution function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Didi detailed search of DegenDuel codebase')
    parser.add_argument('query', type=str, nargs='+', help='The search query')
    parser.add_argument('-n', '--num-results', type=int, default=NUM_RESULTS, 
                        help=f'Number of results to return (default: {NUM_RESULTS})')
    args = parser.parse_args()
    
    # Join query words into a single string
    query = " ".join(args.query)
    
    # Check dependencies first
    if not check_dependencies():
        mock_detailed_search(query, args.num_results)
        return
    
    # Load the index
    logger.info("Loading Didi's knowledge base...")
    try:
        index = load_index()
        
        # Perform search
        detailed_search(index, query, args.num_results)
    except ImportError as e:
        print_colored(f"\nError: {e}", "red")
        print_colored("Didi is missing some required packages.", "yellow")
        mock_detailed_search(query, args.num_results)
    except Exception as e:
        print_colored(f"\nError: {e}", "red")
        print_colored("Didi encountered an error while searching.", "yellow")

if __name__ == "__main__":
    main()