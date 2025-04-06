#\!/usr/bin/env python3
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

# Repository path - where the DegenDuel repository is located
REPO_DIR = Path(os.environ.get("CODE_RAG_REPO_PATH", BASE_DIR / "repo"))

# Database path - where Didi's knowledge base is stored
DB_DIR = Path(os.environ.get("CODE_RAG_DB_PATH", BASE_DIR / "chroma_db"))

# Collection name in ChromaDB - can be customized per model profile
COLLECTION_NAME = os.environ.get("CODE_RAG_COLLECTION_NAME", "degenduel_code")

# Model profile selection - determines which configuration to use
MODEL_PROFILE = os.environ.get("DIDI_MODEL_PROFILE", "default")

# Path to model profiles directory
PROFILES_DIR = BASE_DIR / "model_profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)

# Get profile config file path
PROFILE_CONFIG_PATH = PROFILES_DIR / f"{MODEL_PROFILE}.json"

# Default profiles if no profile exists yet
DEFAULT_PROFILES = {
    "default": {
        "name": "Default (CodeLlama-7B)",
        "llm_model": "codellama/CodeLlama-7b-instruct-hf",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "collection_name": "degenduel_code",
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "context_window": 4096,
        "max_new_tokens": 1024,
        "temperature": 0.1
    },
    "powerful": {
        "name": "Powerful (CodeLlama-34B)",
        "llm_model": "codellama/CodeLlama-34b-instruct-hf",
        "embed_model": "BAAI/bge-large-en-v1.5",
        "collection_name": "degenduel_code_large",
        "chunk_size": 8192,
        "chunk_overlap": 1024,
        "context_window": 8192,
        "max_new_tokens": 2048,
        "temperature": 0.1
    },
    "balanced": {
        "name": "Balanced (Mixtral)",
        "llm_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "embed_model": "BAAI/bge-large-en-v1.5",
        "collection_name": "degenduel_code_balanced",
        "chunk_size": 4096, 
        "chunk_overlap": 512,
        "context_window": 8192,
        "max_new_tokens": 1536,
        "temperature": 0.1
    },
    "gh200": {
        "name": "GH200 (Llama-3-70B)",
        "llm_model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "embed_model": "BAAI/bge-large-en-v1.5",
        "collection_name": "degenduel_code_gh200",
        "chunk_size": 16384,
        "chunk_overlap": 2048,
        "context_window": 32768,
        "max_new_tokens": 4096,
        "temperature": 0.1
    },
    "code": {
        "name": "Code Optimized (DeepSeek-Coder)",
        "llm_model": "deepseek-ai/deepseek-coder-33b-instruct",
        "embed_model": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        "collection_name": "degenduel_code_optimized",
        "chunk_size": 8192,
        "chunk_overlap": 1024,
        "context_window": 16384,
        "max_new_tokens": 2048,
        "temperature": 0.1
    },
    "openchat": {
        "name": "OpenChat 3.5 (Fast & Accurate)",
        "llm_model": "openchat/openchat-3.5-0106",
        "embed_model": "BAAI/bge-large-en-v1.5",
        "collection_name": "degenduel_code_openchat",
        "chunk_size": 4096,
        "chunk_overlap": 512,
        "context_window": 8192,
        "max_new_tokens": 2048,
        "temperature": 0.1
    },
    "llama3": {
        "name": "Llama-3 (8B Balanced)",
        "llm_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "embed_model": "intfloat/e5-large-v2",
        "collection_name": "degenduel_code_llama3",
        "chunk_size": 4096,
        "chunk_overlap": 512,
        "context_window": 8192,
        "max_new_tokens": 2048,
        "temperature": 0.1
    },
    "mistral": {
        "name": "Mistral (7B Fast)",
        "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "collection_name": "degenduel_code_mistral",
        "chunk_size": 4096,
        "chunk_overlap": 512,
        "context_window": 8192,
        "max_new_tokens": 2048,
        "temperature": 0.1
    },
    "phi3": {
        "name": "Phi-3 (Fast & Small)",
        "llm_model": "microsoft/Phi-3-mini-4k-instruct",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "collection_name": "degenduel_code_phi3",
        "chunk_size": 2048,
        "chunk_overlap": 256,
        "context_window": 4096,
        "max_new_tokens": 1024,
        "temperature": 0.1
    },
    "ultra": {
        "name": "UltraRAG (GH200 Optimized)",
        "llm_model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "embed_model": "intfloat/e5-large-v2",
        "collection_name": "degenduel_code_ultra",
        "chunk_size": 32768,
        "chunk_overlap": 4096,
        "context_window": 65536,
        "max_new_tokens": 8192,
        "temperature": 0.1
    }
}

# Create default profile configs if they don't exist
for profile_name, profile_data in DEFAULT_PROFILES.items():
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        with open(profile_path, "w") as f:
            json.dump(profile_data, f, indent=2)

# Load selected profile
if not PROFILE_CONFIG_PATH.exists():
    print(f"Profile {MODEL_PROFILE} not found, creating default config")
    with open(PROFILE_CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_PROFILES["default"], f, indent=2)

# Load the profile
with open(PROFILE_CONFIG_PATH, "r") as f:
    PROFILE = json.load(f)

# Extract configuration from profile
DEFAULT_MODEL_PATH = os.environ.get("CODE_RAG_MODEL_PATH", PROFILE.get("llm_model"))
DEFAULT_EMBED_MODEL = os.environ.get("CODE_RAG_EMBED_MODEL", PROFILE.get("embed_model"))
COLLECTION_NAME = os.environ.get("CODE_RAG_COLLECTION_NAME", PROFILE.get("collection_name", COLLECTION_NAME))
CHUNK_SIZE = PROFILE.get("chunk_size", 1024)
CHUNK_OVERLAP = PROFILE.get("chunk_overlap", 128) 
CONTEXT_WINDOW = PROFILE.get("context_window", 4096)
MAX_NEW_TOKENS = PROFILE.get("max_new_tokens", 1024)
TEMPERATURE = PROFILE.get("temperature", 0.1)

# Available embedding models for testing
EMBEDDING_MODELS = {
    # Original model - general purpose
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    
    # Code-specific model - better for code understanding
    "code": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    
    # Larger, more powerful general model
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    
    # BGE models - excellent for semantic search
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    
    # E5 models - good for general embeddings
    "e5-small": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
    "e5-large": "intfloat/e5-large-v2"
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
    ".sol",  # Solana contracts
    ".rs",   # Rust code
    ".toml", # Config files
    ".yaml", 
    ".yml"
]

# Function to list available profiles
def list_profiles():
    profiles = {}
    for profile_file in PROFILES_DIR.glob("*.json"):
        try:
            with open(profile_file, "r") as f:
                profile_data = json.load(f)
                profile_name = profile_file.stem
                profiles[profile_name] = profile_data.get("name", profile_name)
        except Exception as e:
            print(f"Error loading profile {profile_file}: {e}")
    return profiles

# Function to create a new profile
def create_profile(profile_name, config_data):
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    with open(profile_path, "w") as f:
        json.dump(config_data, f, indent=2)
    return True

# Function to get current profile data
def get_current_profile():
    return {
        "profile_name": MODEL_PROFILE,
        "llm_model": DEFAULT_MODEL_PATH,
        "embed_model": DEFAULT_EMBED_MODEL,
        "collection_name": COLLECTION_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "context_window": CONTEXT_WINDOW,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE
    }
