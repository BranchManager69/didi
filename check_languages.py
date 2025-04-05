#!/usr/bin/env python3
"""Check available languages in tree_sitter_languages."""

import tree_sitter_languages
import inspect

# Print all attributes of the tree_sitter_languages module
print("Attributes of tree_sitter_languages module:")
for attr in dir(tree_sitter_languages):
    if not attr.startswith('__'):
        print(f"- {attr}")

# Try to find any language-related attributes
print("\nTrying to find available languages:")
try:
    # Check if there's a function or attribute that might contain languages
    if hasattr(tree_sitter_languages, 'LANGUAGES'):
        print(tree_sitter_languages.LANGUAGES)
    elif hasattr(tree_sitter_languages, 'languages'):
        print(tree_sitter_languages.languages)
    
    # Try to directly import a few common languages
    common_langs = ['python', 'javascript', 'typescript', 'c', 'cpp', 'java']
    for lang in common_langs:
        try:
            language = getattr(tree_sitter_languages, lang, None)
            if language:
                print(f"Found language: {lang}")
        except Exception as e:
            print(f"Error loading {lang}: {e}")
except Exception as e:
    print(f"Error: {e}")

# Print the core module
if hasattr(tree_sitter_languages, 'core'):
    print("\nAttributes of tree_sitter_languages.core:")
    for attr in dir(tree_sitter_languages.core):
        if not attr.startswith('__'):
            print(f"- {attr}") 