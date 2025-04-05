#!/usr/bin/env python3
"""
Didi A/B Test Demo: Simplified Embedding Model Comparison
A user-friendly wrapper around the main A/B testing system.
"""

import os
import sys
import argparse
import subprocess

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    METRICS_DIR, DEFAULT_EMBED_MODEL, AB_TEST_MODEL
)

def print_colored(text, color=None):
    """Print text with ANSI color codes."""
    colors = {
        'green': '\033[0;32m',
        'blue': '\033[0;34m',
        'yellow': '\033[1;33m',
        'red': '\033[0;31m',
        'reset': '\033[0m'
    }
    
    if color and color in colors:
        print(f"{colors[color]}{text}{colors['reset']}")
    else:
        print(text)

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import importlib.util
        
        required_modules = [
            "llama_index",
            "chromadb",
            "sentence_transformers"
        ]
        
        missing = []
        for module in required_modules:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        
        if missing:
            print_colored("Missing required dependencies:", "red")
            for module in missing:
                print_colored(f"  - {module}", "red")
            print_colored("\nPlease install the missing dependencies with:", "yellow")
            print_colored("pip install " + " ".join(missing), "yellow")
            return False
            
        return True
    except Exception as e:
        print_colored(f"Error checking dependencies: {e}", "red")
        return False

def compare_embeddings_manually(queries):
    """Perform a simplified comparison without the full testing framework."""
    print_colored("Performing simplified comparison", "blue")
    print_colored(f"Model A: {DEFAULT_EMBED_MODEL}", "green")
    print_colored(f"Model B: {AB_TEST_MODEL}", "green")
    
    for query in queries:
        print_colored(f"\nQuery: {query}", "yellow")
        print_colored("Since we can't run the full test, here's what these models would typically do:", "blue")
        print_colored("1. Model A (general purpose) would find documents with similar meaning", "green")
        print_colored("2. Model B (code-specific) would focus on code-related matches", "green")
        print_colored("Example differences:", "yellow")
        print_colored("- Model A might match on semantic similarity", "blue")
        print_colored("- Model B might prioritize code syntax and structure", "blue")
    
    print_colored("\nTo enable full testing, please install the required dependencies.", "yellow")

def run_tests(queries):
    """Run the A/B tests with the given queries."""
    # Ensure metrics directory exists
    if not os.path.exists(METRICS_DIR):
        os.makedirs(METRICS_DIR)
    
    print_colored(f"Running A/B test with models:", "blue")
    print_colored(f"- Model A: {DEFAULT_EMBED_MODEL}", "green")
    print_colored(f"- Model B: {AB_TEST_MODEL}", "green")
    print_colored(f"Testing queries: {', '.join(queries)}\n", "yellow")
    
    # Check if we have the required dependencies
    if not check_dependencies():
        print_colored("\nCannot run full embedding tests due to missing dependencies.", "red")
        compare_embeddings_manually(queries)
        return
    
    # Build the arguments for the main test script
    args = ["python", "scripts/ab_test_embeddings.py", "--queries"]
    args.extend(queries)
    
    # Run the main test script
    try:
        subprocess.run(args, check=True)
        
        # Find the latest summary file
        summaries = sorted(
            [f for f in os.listdir(METRICS_DIR) if f.startswith("embed_test_summary_")],
            reverse=True
        )
        
        if summaries:
            latest_summary = os.path.join(METRICS_DIR, summaries[0])
            print_colored("\nTest Results Summary:", "blue")
            print_colored("-" * 40, "blue")
            
            # Print the summary file
            with open(latest_summary, 'r') as f:
                for line in f:
                    if "Didi A/B Testing Results" in line or "=" * 40 in line:
                        print_colored(line.strip(), "blue")
                    elif "Models Compared:" in line or "Performance Metrics:" in line or "Result Overlap:" in line or "Queries:" in line:
                        print_colored(line.strip(), "yellow")
                    elif "-" * 20 in line:
                        print_colored(line.strip(), "blue")
                    elif "vs" in line and "overlap" in line:
                        print_colored(line.strip(), "green")
                    else:
                        print(line.strip())
            
            print_colored(f"\nFull results available at: {METRICS_DIR}", "green")
        else:
            print_colored("No results found. There may have been an error in the test.", "red")
    
    except subprocess.CalledProcessError as e:
        print_colored(f"\nError running embedding tests: {e}", "red")
        print_colored("Falling back to simplified comparison...", "yellow")
        compare_embeddings_manually(queries)

def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run A/B tests on embedding models for Didi')
    parser.add_argument('queries', nargs='*', help='Queries to test')
    args = parser.parse_args()
    
    # Get queries from arguments or use defaults
    queries = args.queries
    if not queries:
        queries = [
            "websocket connection",
            "user authentication",
            "contest system",
            "wallet integration",
            "loading state handling",
        ]
        print_colored("No queries provided, using default queries.", "yellow")
    
    run_tests(queries)

if __name__ == "__main__":
    main()