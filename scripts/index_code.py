#!/usr/bin/env python3
"""
DegenDuel Codebase Indexing Script for Didi
Builds Didi's knowledge base from the DegenDuel codebase.
"""

import os
import logging
from pathlib import Path
import time
from typing import List, Optional
import glob
import sys

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPO_DIR, DB_DIR, IGNORE_DIRS, INCLUDE_EXTS, DEFAULT_EMBED_MODEL

# LlamaIndex imports
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(repo_path: Path):
    """Load documents from the DegenDuel repository."""
    logger.info(f"Loading documents from {repo_path}")
    
    # Create exclude patterns for directories to ignore
    exclude_patterns = []
    for ignore_dir in IGNORE_DIRS:
        exclude_patterns.append(f"**/{ignore_dir}/**")
    
    # Filter files manually to get only the ones we want to include
    input_files = []
    for ext in INCLUDE_EXTS:
        # Find all files with the given extension
        for file_path in glob.glob(f"{repo_path}/**/*{ext}", recursive=True):
            # Check if the file should be excluded
            if not any(ignore_dir in file_path for ignore_dir in IGNORE_DIRS):
                input_files.append(file_path)
    
    logger.info(f"Found {len(input_files)} files to index")
    
    # Use SimpleDirectoryReader with the input_files parameter
    documents = SimpleDirectoryReader(
        input_files=input_files,
        exclude_hidden=True,
    ).load_data()
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def build_index(documents, db_path: Path, collection_name: str = "degenduel_code"):
    """Build Didi's knowledge base from documents."""
    logger.info("Building Didi's knowledge base...")
    start_time = time.time()
    
    # Initialize ChromaDB client
    db = chromadb.PersistentClient(path=str(db_path))
    
    # Create or get collection
    chroma_collection = db.get_or_create_collection(name=collection_name)
    
    # Initialize vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Configure optimal settings for GH200
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=DEFAULT_EMBED_MODEL,  # Faster embedding model
        max_length=512,  # Increased for code context
        trust_remote_code=True,
    )
    
    # Use token-based text splitter that's optimized for code
    # This splitter doesn't rely on tree-sitter and will work reliably
    Settings.node_parser = TokenTextSplitter(
        chunk_size=1024,  # Roughly 40-50 lines of code
        chunk_overlap=128,  # Decent overlap for context
        separator="\n",  # Split by newlines for code
    )
    
    # Create index - this does the heavy lifting
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    
    end_time = time.time()
    logger.info(f"Knowledge base built in {end_time - start_time:.2f} seconds")
    
    return index

def main():
    """Main execution function."""
    logger.info("Starting to build Didi's knowledge base...")
    
    # Create database directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Verify repo exists
    if not REPO_DIR.exists():
        logger.error(f"DegenDuel repository path {REPO_DIR} does not exist!")
        return
    
    # Load documents
    documents = load_documents(REPO_DIR)
    
    # Build index
    build_index(documents, DB_DIR)
    
    logger.info("Didi's knowledge base is now complete! It's stored in the chroma_db directory.")
    logger.info(f"Total documents indexed: {len(documents)}")

if __name__ == "__main__":
    main() 