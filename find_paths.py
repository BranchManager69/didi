#!/usr/bin/env python3
"""
Script to discover the correct import paths for LlamaIndex modules.
"""

import sys
import pkgutil
import importlib

def find_module_path(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"Found {module_name} at {module.__file__}")
        return True
    except ImportError:
        print(f"Could not import {module_name}")
        return False

# Check for different naming patterns
modules_to_check = [
    # Core modules
    "llama_index.core",
    "llama_index",
    # Vector stores
    "llama_index.vector_stores.chroma",
    "llama_index_vector_stores_chroma",
    "llama_index.vector_stores",
    # Embeddings
    "llama_index.embeddings.huggingface",
    "llama_index_embeddings_huggingface",
    "llama_index.embeddings",
    # LLMs
    "llama_index.llms.huggingface",
    "llama_index_llms_huggingface"
]

for module in modules_to_check:
    find_module_path(module)

print("\nListing all llama_index related packages:")
for pkg in pkgutil.iter_modules():
    if 'llama' in pkg.name:
        print(f"Found package: {pkg.name}")

if __name__ == "__main__":
    pass 