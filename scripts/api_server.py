#!/usr/bin/env python3
"""
Didi HTTP API Server
Provides HTTP API access to Didi's code search and question answering capabilities.

This module implements a Flask-based REST API for Didi, allowing other applications
to programmatically interact with Didi's knowledge base. The API supports semantic
code search, question answering, and system status monitoring.

Key endpoints:
- /health - Health check endpoint
- /api/search - Search code in the repositories
- /api/ask - Ask questions about the codebase
- /api/system/status - Get system component status

The API is designed to be easily extensible with new endpoints and features.
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import Flask
try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Please run: pip install flask flask-cors")
    print("Then restart the API server.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API version
API_VERSION = "1.0.0"

# Optional API key authentication
# Set the DIDI_API_KEYS environment variable to a comma-separated list of API keys
# Example: export DIDI_API_KEYS="key1,key2,key3"
API_KEYS = os.environ.get("DIDI_API_KEYS", "").split(",") if os.environ.get("DIDI_API_KEYS") else []
REQUIRE_AUTH = len(API_KEYS) > 0

# Create Flask app
app = Flask(__name__, static_folder='../public', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Global variables for LLM and query engine
# These are lazily initialized to avoid loading resources until needed
index = None
llm = None
query_engine = None

# Request timing middleware - helps with monitoring API performance
@app.before_request
def start_timer():
    request.start_time = time.time()

@app.after_request
def log_request_info(response):
    # Skip logging for static assets if we had any
    if request.path.startswith('/static/'):
        return response
        
    # Calculate request duration
    duration = time.time() - getattr(request, 'start_time', time.time())
    
    # Log request details
    logger.info(
        f"Request: {request.method} {request.path} | "
        f"Status: {response.status_code} | "
        f"Duration: {duration:.2f}s | "
        f"IP: {request.remote_addr}"
    )
    
    return response

# Authentication middleware if API keys are configured
if REQUIRE_AUTH:
    @app.before_request
    def authenticate():
        # Skip auth for health check endpoint
        if request.path == '/health':
            return None
            
        # Get API key from header or query parameter
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key or api_key not in API_KEYS:
            return jsonify({
                "error": "Unauthorized. Please provide a valid API key in the X-API-Key header."
            }), 401

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    
    This endpoint provides basic information about the API service status and
    can be used for monitoring and connectivity testing. It doesn't require 
    authentication, even if API keys are enabled.
    
    Returns:
        JSON with status information
    """
    return jsonify({
        "status": "ok",
        "service": "didi-api",
        "version": API_VERSION,
        "timestamp": int(time.time()),
        "auth_required": REQUIRE_AUTH,
    })

@app.route('/api/search', methods=['POST'])
def search_code():
    """
    Endpoint for semantic code search.
    
    This endpoint allows finding relevant code files and snippets based on
    natural language queries using semantic search. It can return either quick
    search results or more detailed code snippets depending on the "detailed" parameter.
    
    Expected JSON payload:
    {
        "query": "string",           # required - The search query string
        "limit": int,                # optional - Maximum number of results (default: 10)
        "detailed": boolean          # optional - Return detailed code snippets (default: false)
    }
    
    Returns:
        JSON with search results including file paths, relevance scores, and code snippets
        HTTP 400 if the query parameter is missing
        HTTP 500 if an error occurs during search
    """
    start_time = time.time()
    
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must contain valid JSON data"}), 400
            
        if 'query' not in data:
            return jsonify({"error": "Query parameter is required"}), 400
            
        # Extract parameters
        query = data['query']
        limit = int(data.get('limit', 10))
        detailed = bool(data.get('detailed', False))
        
        logger.info(f"Search request: query='{query}', limit={limit}, detailed={detailed}")
        
        # Import search functionality here to avoid circular imports
        if detailed:
            # Detailed search provides more context for each result
            from scripts.simple_search import perform_search
            results = perform_search(query, limit=limit, return_json=True)
        else:
            # Standard search is faster and provides basic results
            from scripts.search_code import perform_search
            results = perform_search(query, limit=limit, return_json=True)
        
        # Add metadata to the response
        if isinstance(results, dict) and 'error' not in results:
            results['query_time'] = f"{time.time() - start_time:.2f}s"
            
        return jsonify(results)
    
    except ValueError as e:
        logger.warning(f"Invalid parameter in search request: {e}")
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Endpoint for asking questions about the codebase.
    
    This endpoint enables natural language questions about the codebase and returns
    AI-generated answers based on relevant code snippets retrieved from the repositories.
    It uses a retrieval-augmented generation (RAG) system to ground answers in the code.
    
    Expected JSON payload:
    {
        "question": "string",        # required - The question about the codebase
        "include_sources": boolean,  # optional - Include source references in response (default: true)
        "model_params": {            # optional - Advanced parameters for the LLM
            "temperature": float,    # optional - Controls randomness (0.0-1.0)
            "max_tokens": int,       # optional - Maximum response length
            "stream": boolean        # optional - Stream the response (NOT IMPLEMENTED)
        }
    }
    
    Returns:
        JSON with the answer and optionally source code references
        HTTP 400 if the question parameter is missing
        HTTP 500 if an error occurs during processing
    """
    global index, llm, query_engine
    start_time = time.time()
    
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must contain valid JSON data"}), 400
            
        if 'question' not in data:
            return jsonify({"error": "Question parameter is required"}), 400
        
        # Extract parameters
        question = data['question']
        include_sources = bool(data.get('include_sources', True))
        
        # Extract advanced model parameters if provided
        model_params = data.get('model_params', {})
        temperature = model_params.get('temperature')
        max_tokens = model_params.get('max_tokens')
        stream = bool(model_params.get('stream', False))
        
        # Log request
        logger.info(f"Question: {question}")
        if model_params:
            logger.info(f"Model params: {model_params}")
        
        # Initialize LLM and query engine if needed
        if index is None or llm is None or query_engine is None:
            from config import DEFAULT_MODEL_PATH
            logger.info(f"Initializing query engine with model: {DEFAULT_MODEL_PATH}")
            
            # Try to use enhanced query if available
            try:
                from scripts.enhanced_query import load_index, setup_llm, setup_hybrid_retriever, setup_query_engine
                
                logger.info("Initializing enhanced query engine...")
                index = load_index()
                llm = setup_llm()
                
                # Apply custom parameters if provided
                if temperature is not None or max_tokens is not None:
                    if hasattr(llm, 'generate_kwargs'):
                        if temperature is not None:
                            llm.generate_kwargs['temperature'] = float(temperature)
                        if max_tokens is not None:
                            llm.generate_kwargs['max_new_tokens'] = int(max_tokens)
                
                retriever = setup_hybrid_retriever(index)
                query_engine = setup_query_engine(retriever, llm)
                logger.info("Enhanced query engine initialized")
                
            except (ImportError, Exception) as e:
                logger.warning(f"Could not initialize enhanced query engine: {e}")
                logger.info("Falling back to standard query engine")
                
                from scripts.query_code import load_index, setup_llm, setup_retriever, setup_query_engine
                
                index = load_index()
                llm = setup_llm()
                
                # Apply custom parameters if provided
                if temperature is not None or max_tokens is not None:
                    if hasattr(llm, 'generate_kwargs'):
                        if temperature is not None:
                            llm.generate_kwargs['temperature'] = float(temperature)
                        if max_tokens is not None:
                            llm.generate_kwargs['max_new_tokens'] = int(max_tokens)
                
                retriever = setup_retriever(index)
                query_engine = setup_query_engine(retriever, llm)
        
        # Streaming not yet implemented - warn if requested
        if stream:
            logger.warning("Streaming responses not yet implemented, returning full response")
        
        # Execute query
        logger.info(f"Processing question with query engine...")
        response = query_engine.query(question)
        
        # Format response
        result = {
            "answer": str(response),
            "question": question,
            "query_time": f"{time.time() - start_time:.2f}s"
        }
        
        # Include sources if requested
        if include_sources and hasattr(response, 'source_nodes') and response.source_nodes:
            sources = []
            for i, node in enumerate(response.source_nodes):
                score = node.score if hasattr(node, 'score') else None
                
                source = {
                    "id": i + 1,  # 1-based index for readability
                    "file_path": node.metadata.get('file_path', 'Unknown'),
                    "repo_name": node.metadata.get('repo_name', 'Unknown'),
                    "relevance": round(score, 4) if score is not None else None,
                    "content": node.text[:500] + ("..." if len(node.text) > 500 else "")
                }
                sources.append(source)
            
            result["sources"] = sources
            result["sources_count"] = len(sources)
        
        logger.info(f"Question answered in {time.time() - start_time:.2f}s")
        return jsonify(result)
    
    except ValueError as e:
        logger.warning(f"Invalid parameter in question request: {e}")
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """
    Get the status of Didi's system components.
    
    This endpoint provides information about the system's database, repositories,
    models, and runtime state. It can be used for monitoring and diagnostics.
    
    Request parameters (optional query parameters):
    - detailed: boolean - Include detailed information about repositories
    
    Returns:
        JSON with status information about the system components
        HTTP 500 if an error occurs during status checking
    """
    try:
        # Import config values
        from config import (
            DB_DIR, REPOS_DIR, MODEL_CACHE_DIR, COLLECTION_NAME, 
            REPOS_CONFIG_FILE, DEFAULT_EMBED_MODEL, DEFAULT_MODEL_PATH
        )
        
        # Check if detailed info is requested
        detailed = request.args.get('detailed', '').lower() in ('true', '1', 'yes')
        
        # Check database status
        db_exists = DB_DIR.exists()
        db_size = "0" if not db_exists else os.popen(f"du -sh {DB_DIR} | cut -f1").read().strip()
        
        # Get collection information if available
        collection_info = {}
        if db_exists:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(DB_DIR))
                try:
                    collection = client.get_collection(name=COLLECTION_NAME)
                    count = collection.count()
                    collection_info = {
                        "name": COLLECTION_NAME,
                        "doc_count": count
                    }
                except ValueError:
                    collection_info = {"error": f"Collection '{COLLECTION_NAME}' not found"}
            except Exception as e:
                collection_info = {"error": f"Could not access ChromaDB: {str(e)}"}
        
        # Check repositories
        repos_config = {}
        if REPOS_CONFIG_FILE.exists():
            with open(REPOS_CONFIG_FILE, 'r') as f:
                repos_config = json.load(f)
        
        active_repos = {k: v for k, v in repos_config.items() if v.get('enabled', True)}
        
        # Check models
        models_dir_exists = MODEL_CACHE_DIR.exists()
        models_size = "0" if not models_dir_exists else os.popen(f"du -sh {MODEL_CACHE_DIR} | cut -f1").read().strip()
        
        # Check if API is initialized (LLM loaded)
        api_initialized = index is not None and llm is not None and query_engine is not None
        
        # Basic status response
        status = {
            "api": {
                "version": API_VERSION,
                "initialized": api_initialized,
                "timestamp": int(time.time()),
                "uptime": time.time() - app.start_time if hasattr(app, 'start_time') else None
            },
            "database": {
                "exists": db_exists,
                "path": str(DB_DIR),
                "size": db_size,
                "collection": collection_info
            },
            "repositories": {
                "total": len(repos_config),
                "active": len(active_repos),
                "names": [v.get('name', k) for k, v in active_repos.items()]
            },
            "models": {
                "exists": models_dir_exists,
                "path": str(MODEL_CACHE_DIR),
                "size": models_size,
                "llm": DEFAULT_MODEL_PATH,
                "embedding": DEFAULT_EMBED_MODEL
            }
        }
        
        # Add detailed repository information if requested
        if detailed:
            # Gather more information about each repository
            detailed_repos = []
            for key, repo in active_repos.items():
                repo_path = Path(repo.get('path', ''))
                repo_info = {
                    "key": key,
                    "name": repo.get('name', key),
                    "description": repo.get('description', ''),
                    "path": str(repo_path),
                    "exists": repo_path.exists(),
                }
                
                # Get repo stats if it exists
                if repo_path.exists():
                    # Count files by type
                    ts_files = len(list(repo_path.glob('**/*.ts*')))
                    js_files = len(list(repo_path.glob('**/*.js*')))
                    py_files = len(list(repo_path.glob('**/*.py')))
                    md_files = len(list(repo_path.glob('**/*.md')))
                    
                    repo_info["stats"] = {
                        "typescript_files": ts_files,
                        "javascript_files": js_files,
                        "python_files": py_files,
                        "markdown_files": md_files,
                        "total_files": ts_files + js_files + py_files + md_files
                    }
                
                detailed_repos.append(repo_info)
            
            status["repositories"]["details"] = detailed_repos
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

# Serve static files from public directory
@app.route('/')
def index():
    """Serve the frontend application."""
    return app.send_static_file('index.html')

# Add API_REFERENCE.md route
@app.route('/API_REFERENCE.md')
def api_reference():
    """Serve the API reference documentation."""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'API_REFERENCE.md'), 'r') as f:
            content = f.read()
            return Response(content, mimetype='text/markdown')
    except Exception as e:
        logger.error(f"Error serving API reference: {e}")
        return "API Reference not found", 404

# Add a custom 404 error handler
@app.errorhandler(404)
def not_found(e):
    """Custom 404 error handler with API information."""
    # Check if the request is for an API endpoint
    if request.path.startswith('/api/'):
        return jsonify({
            "error": "Endpoint not found",
            "message": "The requested URL was not found on the server.",
            "api_info": {
                "version": API_VERSION,
                "available_endpoints": [
                    {"method": "GET", "path": "/health", "description": "Health check"},
                    {"method": "POST", "path": "/api/search", "description": "Search code"},
                    {"method": "POST", "path": "/api/ask", "description": "Ask questions"},
                    {"method": "GET", "path": "/api/system/status", "description": "Get system status"}
                ],
                "documentation": "See API_REFERENCE.md for full documentation"
            }
        }), 404
    
    # For non-API routes, redirect to frontend
    return redirect('/')

# Add a route for creating custom endpoints dynamically
@app.route('/api/custom', methods=['POST'])
def custom_endpoint():
    """
    Flexible endpoint for custom operations.
    
    This endpoint allows combining multiple operations in a single request or
    performing special operations not covered by the standard endpoints.
    
    Expected JSON payload:
    {
        "operation": "string",       # required - Operation to perform
        "params": object,            # required - Operation-specific parameters
        "context": object            # optional - Additional context information
    }
    
    Supported operations:
    - "search_and_ask": Perform a search and then ask a question about the results
    - "batch": Execute multiple operations in a single request
    
    Returns:
        JSON with the operation result
        HTTP 400 if the operation is invalid or parameters are missing
        HTTP 500 if an error occurs during processing
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must contain valid JSON data"}), 400
            
        if 'operation' not in data:
            return jsonify({"error": "Operation parameter is required"}), 400
            
        operation = data['operation']
        params = data.get('params', {})
        
        # Log the custom request
        logger.info(f"Custom operation: {operation}")
        
        # Handle different operation types
        if operation == "search_and_ask":
            # Validate required parameters
            if 'search_query' not in params or 'question' not in params:
                return jsonify({"error": "search_query and question parameters are required"}), 400
                
            # First perform a search
            from scripts.search_code import perform_search
            search_results = perform_search(
                params['search_query'], 
                limit=params.get('limit', 10), 
                return_json=True
            )
            
            # Then format the question with context from search results
            context = "Based on the following code:\n\n"
            if 'results' in search_results:
                for i, result in enumerate(search_results['results'][:3]):  # Use top 3 results
                    context += f"File: {result['file_path']}\n"
                    context += result['snippet'] + "\n\n"
                    
            # Send the enhanced question to the query engine
            if index is None or llm is None or query_engine is None:
                # Initialize if needed using the /api/ask implementation
                return ask_question()
                
            # Format the final question with context
            final_question = context + "\n" + params['question']
            response = query_engine.query(final_question)
            
            # Return combined results
            return jsonify({
                "search_results": search_results,
                "answer": str(response),
                "operation": "search_and_ask"
            })
            
        elif operation == "batch":
            # Validate the operations array
            if 'operations' not in params or not isinstance(params['operations'], list):
                return jsonify({"error": "operations array is required for batch operation"}), 400
                
            # Process each operation in the batch
            results = []
            for op in params['operations']:
                if 'type' not in op:
                    results.append({"error": "Operation type is required"})
                    continue
                    
                # Handle different operation types
                if op['type'] == 'search':
                    from scripts.search_code import perform_search
                    search_results = perform_search(
                        op.get('query', ''), 
                        limit=op.get('limit', 10), 
                        return_json=True
                    )
                    results.append({"type": "search", "results": search_results})
                    
                elif op['type'] == 'status':
                    # Reuse the system_status endpoint
                    status_result = json.loads(system_status().data)
                    results.append({"type": "status", "results": status_result})
                    
                else:
                    results.append({"type": op['type'], "error": "Unsupported operation type"})
            
            return jsonify({
                "operation": "batch",
                "results": results,
                "count": len(results)
            })
            
        else:
            return jsonify({"error": f"Unsupported operation: {operation}"}), 400
            
    except Exception as e:
        logger.error(f"Error in custom endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

def main():
    """Main function to start the API server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start Didi HTTP API server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes (only used with gunicorn)')
    parser.add_argument('--use-gunicorn', action='store_true', help='Use gunicorn for production deployment')
    
    args = parser.parse_args()
    
    # Store startup time
    app.start_time = time.time()
    
    # Print startup banner
    print("\n" + "=" * 60)
    print(" Didi HTTP API Server ".center(60))
    print("=" * 60)
    print(f"\nStarting server on {args.host}:{args.port}")
    
    if REQUIRE_AUTH:
        print(f"API key authentication enabled ({len(API_KEYS)} keys configured)")
    else:
        print("API key authentication disabled")
        
    print("\nAvailable endpoints:")
    print("  GET  /health               - Health check")
    print("  POST /api/search           - Search code")
    print("  POST /api/ask              - Ask questions")
    print("  GET  /api/system/status    - Get system status")
    print("  POST /api/custom           - Custom operations endpoint")
    print("\nDocumentation available in API_REFERENCE.md")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        # Try to use gunicorn if requested
        if args.use_gunicorn:
            try:
                from gunicorn.app.base import BaseApplication
                
                class GunicornApplication(BaseApplication):
                    def __init__(self, app, options=None):
                        self.options = options or {}
                        self.application = app
                        super().__init__()
                    
                    def load_config(self):
                        for key, value in self.options.items():
                            if key in self.cfg.settings and value is not None:
                                self.cfg.set(key, value)
                    
                    def load(self):
                        return self.application
                
                # Configure Gunicorn options
                options = {
                    'bind': f"{args.host}:{args.port}",
                    'workers': args.workers,
                    'worker_class': 'sync',
                    'accesslog': '-',
                    'errorlog': '-',
                    'loglevel': 'info' if not args.debug else 'debug',
                }
                
                print(f"Starting with gunicorn ({args.workers} workers)")
                GunicornApplication(app, options).run()
                
            except ImportError:
                print("Gunicorn not installed. Install with 'pip install gunicorn' for production use.")
                print("Falling back to Flask's built-in server (not recommended for production).")
                app.run(host=args.host, port=args.port, debug=args.debug)
        else:
            # Use Flask's built-in server
            app.run(host=args.host, port=args.port, debug=args.debug)
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()