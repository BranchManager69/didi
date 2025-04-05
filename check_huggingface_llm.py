#!/usr/bin/env python3
"""Check HuggingFaceLLM parameters."""

from llama_index.llms.huggingface import HuggingFaceLLM
import inspect

# Print the signature of HuggingFaceLLM
print("HuggingFaceLLM signature:")
print(inspect.signature(HuggingFaceLLM.__init__))

# Print the docstring for HuggingFaceLLM
print("\nHuggingFaceLLM docstring:")
print(HuggingFaceLLM.__init__.__doc__) 