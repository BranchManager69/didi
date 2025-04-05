#!/usr/bin/env python3
"""Check SimpleDirectoryReader options."""

from llama_index.core import SimpleDirectoryReader
import inspect

# Print the signature of SimpleDirectoryReader
print(inspect.signature(SimpleDirectoryReader))

# Print all available parameters
print("\nAvailable parameters:")
for param_name, param in inspect.signature(SimpleDirectoryReader.__init__).parameters.items():
    if param_name != 'self':
        print(f"- {param_name}: {param.default}") 