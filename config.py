#!/usr/bin/env python3
"""
Didi Configuration

This module provides centralized configuration for Didi, DegenDuel's AI Assistant,
including paths and settings that might need to be changed when deploying
in different environments.
"""

import os
from pathlib import Path

# Base directory - can be overridden with CODE_RAG_PATH environment variable
BASE_DIR = Path(os.environ.get("CODE_RAG_PATH", os.path.dirname(os.path.abspath(__file__))))

# Repository path - where the DegenDuel repository is located
REPO_DIR = Path(os.environ.get("CODE_RAG_REPO_PATH", BASE_DIR / "repo"))

# Database path - where Didi's knowledge base is stored
DB_DIR = Path(os.environ.get("CODE_RAG_DB_PATH", BASE_DIR / "chroma_db"))

# Collection name in ChromaDB
COLLECTION_NAME = os.environ.get("CODE_RAG_COLLECTION_NAME", "degenduel_code")

# Default model path for Didi's brain
DEFAULT_MODEL_PATH = os.environ.get("CODE_RAG_MODEL_PATH", "codellama/CodeLlama-7b-instruct-hf")

# Embedding models - multiple models for A/B testing
# Default embedding model can be overridden with CODE_RAG_EMBED_MODEL environment variable
DEFAULT_EMBED_MODEL = os.environ.get("CODE_RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Available embedding models for testing
EMBEDDING_MODELS = {
    # Original model - general purpose
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    
    # Code-specific model - better for code understanding
    "code": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    
    # Larger, more powerful general model
    "mpnet": "sentence-transformers/all-mpnet-base-v2"
}

# A/B testing configuration
# Set to True to enable A/B testing of embedding models
ENABLE_AB_TESTING = os.environ.get("CODE_RAG_AB_TESTING", "False").lower() == "true"

# If A/B testing is enabled, use this as the second model
AB_TEST_MODEL = os.environ.get("CODE_RAG_AB_TEST_MODEL", EMBEDDING_MODELS["code"])

# A/B testing metrics collection directory
METRICS_DIR = BASE_DIR / "metrics"

# Directories to ignore during indexing
IGNORE_DIRS = [
    ".git", 
    "node_modules", 
    "dist", 
    "dist-dev", 
    ".next", 
    "coverage", 
    "out",
]

# File extensions to include during indexing
INCLUDE_EXTS = [
    ".ts", 
    ".tsx", 
    ".js", 
    ".jsx", 
    ".css", 
    ".scss", 
    ".json", 
    ".md", 
    ".html"
] 