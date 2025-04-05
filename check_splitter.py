#!/usr/bin/env python3
"""Check CodeSplitter options."""

from llama_index.core.node_parser import CodeSplitter
import inspect

# Print the signature of CodeSplitter
print(inspect.signature(CodeSplitter))

# Print all available parameters
print("\nAvailable parameters:")
for param_name, param in inspect.signature(CodeSplitter.__init__).parameters.items():
    if param_name != 'self':
        print(f"- {param_name}: {param.default}") 