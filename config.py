#!/usr/bin/env python3
"""
Didi Configuration

This module provides centralized configuration for Didi, DegenDuel's AI Assistant,
including paths and settings that might need to be changed when deploying
in different environments.
"""

import os
import json
from pathlib import Path

# Base directory - can be overridden with CODE_RAG_PATH environment variable
BASE_DIR = Path(os.environ.get("CODE_RAG_PATH", os.path.dirname(os.path.abspath(__file__))))

# Repositories directory - where all codebases are located
# Use persistent storage for repos
REPOS_DIR = Path(os.environ.get("CODE_RAG_REPOS_PATH", "/home/ubuntu/degenduel-gpu/repos"))

# Database path - where Didi's knowledge base is stored
# Use persistent storage for the database
DB_DIR = Path(os.environ.get("CODE_RAG_DB_PATH", "/home/ubuntu/degenduel-gpu/data/chroma_db"))

# Collection name in ChromaDB
COLLECTION_NAME = os.environ.get("CODE_RAG_COLLECTION_NAME", "multi_codebase")

# Default model path for Didi's brain
# Use local path for models to avoid redownloading
MODEL_CACHE_DIR = Path(os.environ.get("HF_HOME", "/home/ubuntu/degenduel-gpu/models"))
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
    ".html",
    ".py",
    ".rs",
    ".go",
    ".java",
    ".rb",
    ".php",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
]

# Repositories configuration
# Use persistent storage for repos config
REPOS_CONFIG_FILE = Path(os.environ.get("CODE_RAG_CONFIG_PATH", "/home/ubuntu/degenduel-gpu/config/repos_config.json"))

# Default repositories if no config file exists
DEFAULT_REPOS = {
    "degenduel": {
        "name": "DegenDuel",
        "description": "The main DegenDuel application codebase",
        "path": str(REPOS_DIR / "degenduel"),
        "git_url": "",  # Leave empty if already cloned
        "enabled": True
    }
}

# Load repositories configuration or create default
def load_repos_config():
    if REPOS_CONFIG_FILE.exists():
        with open(REPOS_CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        # Create default config
        with open(REPOS_CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_REPOS, f, indent=2)
        return DEFAULT_REPOS

# Active repositories (enabled in config)
REPOS = load_repos_config()
ACTIVE_REPOS = {k: v for k, v in REPOS.items() if v.get('enabled', True)}