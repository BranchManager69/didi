#!/usr/bin/env python3
"""
Didi Parallel Codebase Indexing Script
Builds Didi's knowledge base from multiple codebases using parallel processing.
"""

import os
import logging
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
import glob
import sys
import json
import concurrent.futures
import multiprocessing

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPOS_DIR, DB_DIR, IGNORE_DIRS, INCLUDE_EXTS, DEFAULT_EMBED_MODEL, COLLECTION_NAME, ACTIVE_REPOS, MODEL_CACHE_DIR

# LlamaIndex imports
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine optimal number of workers
cpu_count = multiprocessing.cpu_count()
OPTIMAL_WORKERS = min(32, max(8, cpu_count - 4))  # Leave some cores for system operations
logger.info(f"Using {OPTIMAL_WORKERS} workers for parallel processing")

def load_file(file_path: str, repo_key: str, repo_info: Dict[str, Any]) -> Document:
    """Load a single file and return a Document with proper metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            try:
                content = f.read()
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                content = f"Error: Could not read file due to {str(e)}"
        
        # Create relative path for better identification
        repo_path = Path(repo_info['path'])
        relative_path = Path(file_path).relative_to(repo_path)
        
        # Create metadata
        metadata = {
            "file_path": str(file_path),
            "relative_path": str(relative_path),
            "repo_key": repo_key,
            "repo_name": repo_info["name"],
            "file_extension": Path(file_path).suffix,
        }
        
        return Document(text=content, metadata=metadata)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def find_files_in_repo(repo_path: Path) -> List[str]:
    """Find all files to index in a repository."""
    input_files = []
    for ext in INCLUDE_EXTS:
        # Find all files with the given extension
        for file_path in glob.glob(f"{repo_path}/**/*{ext}", recursive=True):
            # Check if the file should be excluded
            if not any(ignore_dir in file_path for ignore_dir in IGNORE_DIRS):
                input_files.append(file_path)
    return input_files

def load_documents_from_repo(repo_key: str, repo_info: Dict[str, Any]) -> List[Document]:
    """Load documents from a single repository using parallel processing."""
    repo_path = Path(repo_info["path"])
    logger.info(f"Loading documents from {repo_info['name']} ({repo_path})")
    
    # Find all files to index
    input_files = find_files_in_repo(repo_path)
    logger.info(f"Found {len(input_files)} files to index in {repo_info['name']}")
    
    if not input_files:
        return []
    
    # Load files in parallel
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=OPTIMAL_WORKERS) as executor:
        future_to_file = {
            executor.submit(load_file, file_path, repo_key, repo_info): file_path 
            for file_path in input_files
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                doc = future.result()
                if doc:
                    documents.append(doc)
                
                # Show progress
                if (i + 1) % 100 == 0 or i + 1 == len(input_files):
                    logger.info(f"Processed {i + 1}/{len(input_files)} files in {repo_info['name']}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents from {repo_info['name']}")
    return documents

def load_all_documents() -> List[Document]:
    """Load documents from all active repositories in parallel."""
    all_documents = []
    
    # Process each repository in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(ACTIVE_REPOS), 4)) as executor:
        future_to_repo = {}
        for repo_key, repo_info in ACTIVE_REPOS.items():
            # Check if repository exists
            repo_path = Path(repo_info["path"])
            if not repo_path.exists():
                logger.warning(f"Repository path {repo_path} does not exist! Skipping.")
                continue
            
            # Submit task
            future = executor.submit(load_documents_from_repo, repo_key, repo_info)
            future_to_repo[future] = repo_key
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_repo):
            repo_key = future_to_repo[future]
            try:
                repo_documents = future.result()
                all_documents.extend(repo_documents)
                logger.info(f"Added {len(repo_documents)} documents from {repo_key}")
            except Exception as e:
                logger.error(f"Error processing repository {repo_key}: {e}")
    
    logger.info(f"Loaded {len(all_documents)} documents from {len(ACTIVE_REPOS)} repositories")
    return all_documents

def build_index(documents: List[Document], db_path: Path, collection_name: str = COLLECTION_NAME):
    """Build Didi's knowledge base from documents."""
    logger.info("Building Didi's knowledge base...")
    start_time = time.time()
    
    # Make sure the database directory exists and has proper permissions
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize ChromaDB client
    db = chromadb.PersistentClient(path=str(db_path))
    
    # First try to list collections to check if client is working
    try:
        existing_collections = db.list_collections()
        logger.info(f"Found {len(existing_collections)} existing collections")
    except Exception as e:
        logger.error(f"Error accessing ChromaDB: {e}")
        logger.info("Trying to reset ChromaDB directory...")
        # Try to reset the directory
        import shutil
        backup_path = str(db_path) + "_backup_" + time.strftime("%Y%m%d%H%M%S")
        if os.path.exists(str(db_path)):
            shutil.move(str(db_path), backup_path)
            logger.info(f"Moved existing database to {backup_path}")
        os.makedirs(db_path, exist_ok=True)
        # Reinitialize client
        db = chromadb.PersistentClient(path=str(db_path))
    
    # Try to delete collection if it exists
    try:
        collection_exists = False
        for col in db.list_collections():
            if col.name == collection_name:
                collection_exists = True
                break
                
        if collection_exists:
            logger.info(f"Deleting existing collection: {collection_name}")
            db.delete_collection(name=collection_name)
    except Exception as e:
        logger.warning(f"Error checking/deleting collection: {e}")
    
    # Create new collection
    try:
        chroma_collection = db.create_collection(name=collection_name)
        logger.info(f"Created new collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise
    
    # Initialize vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Configure optimal settings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=DEFAULT_EMBED_MODEL,
        max_length=512,  # Increased for code context
        trust_remote_code=True,
        cache_folder=str(MODEL_CACHE_DIR),  # Use persistent cache
    )
    
    # Use token-based text splitter that's optimized for code
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
    
    # Save repository metadata
    repos_meta_file = db_path / "repos_metadata.json"
    with open(repos_meta_file, 'w') as f:
        json.dump(ACTIVE_REPOS, f, indent=2)
    
    return index

def main():
    """Main execution function."""
    logger.info("Starting to build Didi's knowledge base...")
    
    # Create database directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Verify we have active repos
    if not ACTIVE_REPOS:
        logger.error("No active repositories configured!")
        return
    
    # Load documents from all repositories
    documents = load_all_documents()
    
    if not documents:
        logger.error("No documents loaded! Check your repository paths and file patterns.")
        return
    
    # Build index
    build_index(documents, DB_DIR)
    
    logger.info("Didi's knowledge base is now complete! It's stored in the chroma_db directory.")
    logger.info(f"Total documents indexed: {len(documents)}")
    
    # Print repository stats
    repo_counts = {}
    for doc in documents:
        repo_key = doc.metadata.get("repo_key", "unknown")
        repo_counts[repo_key] = repo_counts.get(repo_key, 0) + 1
    
    logger.info("Documents per repository:")
    for repo_key, count in repo_counts.items():
        repo_name = ACTIVE_REPOS.get(repo_key, {}).get("name", repo_key)
        logger.info(f"  - {repo_name}: {count} documents")

if __name__ == "__main__":
    main()