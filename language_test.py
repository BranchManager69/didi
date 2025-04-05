#!/usr/bin/env python3
"""Test tree-sitter language availability"""

import os
import sys
import importlib.util
import importlib.metadata
import tree_sitter
import tree_sitter_languages
import inspect
import pkgutil

try:
    tree_sitter_version = importlib.metadata.version('tree-sitter')
except:
    tree_sitter_version = "Unknown"
    
try:
    tree_sitter_languages_version = importlib.metadata.version('tree-sitter-languages')
except:
    tree_sitter_languages_version = "Unknown"

print(f"Tree-sitter version: {tree_sitter_version}")
print(f"Tree-sitter-languages version: {tree_sitter_languages_version}")

print("\nTree-sitter-languages location:")
print(tree_sitter_languages.__file__)

print("\nTree-sitter-languages package contents:")
for module_info in pkgutil.iter_modules(tree_sitter_languages.__path__):
    print(f"- {module_info.name}")

print("\nChecking tree-sitter language files:")
languages_dir = os.path.dirname(tree_sitter_languages.__file__)
for root, dirs, files in os.walk(languages_dir):
    for filename in files:
        if filename.endswith('.so') or filename.endswith('.dylib') or filename.endswith('.dll'):
            print(f"- {filename}")

# Try loading basic languages
print("\nTrying to access common languages:")
try:
    # Try using the get_language function
    from tree_sitter_languages.core import get_language
    common_languages = ["c", "cpp", "go", "java", "javascript", "python", "rust", "typescript"]
    for lang in common_languages:
        try:
            parser = tree_sitter.Parser()
            lang_obj = get_language(lang)
            parser.set_language(lang_obj)
            print(f"✓ Successfully loaded language: {lang}")
        except Exception as e:
            print(f"✗ Failed to load language {lang}: {e}")
except Exception as e:
    print(f"Error accessing languages: {e}") 