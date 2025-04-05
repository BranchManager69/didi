#!/usr/bin/env python3
"""
Debug script to find where the OpenAI dependency is coming from.
"""

import sys
import logging
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Find and report OpenAI dependencies."""
    try:
        # Load the main modules with debug tracing
        from llama_index.core.response_synthesizers import TreeSummarize
        logger.info("Successfully imported TreeSummarize")
        
        # Try to create a TreeSummarize instance
        ts = TreeSummarize(llm=None, verbose=True)
        logger.info("Successfully created TreeSummarize instance without LLM")
        
        # Import other key components
        from llama_index.core.indices.vector_store import VectorStoreIndex
        logger.info("Successfully imported VectorStoreIndex")
        
        from llama_index.core.retrievers import VectorIndexRetriever
        logger.info("Successfully imported VectorIndexRetriever")
        
        from llama_index.core.query_engine import RetrieverQueryEngine
        logger.info("Successfully imported RetrieverQueryEngine")
        
        # Try to build the query engine without OpenAI
        from llama_index.llms.huggingface import HuggingFaceLLM
        logger.info("Successfully imported HuggingFaceLLM")
        
        # Try creating a dummy query engine to see where it fails
        from llama_index.core.schema import Document
        docs = [Document(text="Test document")]
        
        # Try importing each response synthesizer type
        logger.info("Trying to import all response synthesizer types...")
        from llama_index.core.response_synthesizers import (
            CompactAndRefine,
            TreeSummarize,
            Refine,
            CompactAndRefine,
            SimpleResponseBuilder,
            ResponseMode,
        )
        logger.info("Successfully imported all response synthesizer types")
        
        # Print package versions
        import llama_index
        logger.info(f"Llama-index version: {llama_index.__version__}")
        
        # Check what happens inside TreeSummarize
        logger.info("Inspecting TreeSummarize initialization...")
        ts_instance = TreeSummarize(llm=None)
        logger.info(f"TreeSummarize instance: {ts_instance}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()