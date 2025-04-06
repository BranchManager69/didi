# Didi HTTP API Reference

This document provides comprehensive documentation for Didi's HTTP API, which allows programmatic access to Didi's code search and question answering capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Server Setup](#server-setup)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
   - [Health Check](#health-check)
   - [Code Search](#code-search)
   - [Question Answering](#question-answering)
   - [System Status](#system-status)
5. [Response Formats](#response-formats)
6. [Error Handling](#error-handling)
7. [Extending the API](#extending-the-api)
8. [Example Implementations](#example-implementations)

## Overview

Didi's HTTP API provides RESTful access to the AI assistant's capabilities, enabling integration with other services and applications. The API is built with Flask and follows standard REST conventions.

Key features:
- JSON-based request and response format
- Comprehensive error handling
- Cross-Origin Resource Sharing (CORS) support
- Stateless architecture
- Extensible design

## Server Setup

### Starting the API Server

```bash
# Start with default settings (host: 0.0.0.0, port: 8000)
./didi.sh api

# Start on a specific port
./didi.sh api --port 3000

# Enable debug mode
./didi.sh api --debug
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--host` | Host address to bind to | `0.0.0.0` |
| `--port` | Port to listen on | `8000` |
| `--debug` | Enable debug mode | `False` |

### Docker Configuration

When running in Docker, you can expose the API port:

```yaml
# In docker-compose.yml
services:
  didi:
    # ...
    ports:
      - "8000:8000"
```

## Authentication

The current implementation does not include authentication. For production use, consider implementing one of the following:

1. API Key authentication:
   ```python
   # Example implementation in api_server.py
   API_KEYS = os.environ.get("DIDI_API_KEYS", "").split(",")
   
   @app.before_request
   def check_auth():
       if request.endpoint != 'health_check':  # Skip auth for health checks
           api_key = request.headers.get("X-API-Key")
           if not api_key or api_key not in API_KEYS:
               return jsonify({"error": "Unauthorized"}), 401
   ```

2. JWT authentication
3. OAuth2 integration

## API Endpoints

### Health Check

**Endpoint:** `GET /health`

Simple health check endpoint to verify the API is running.

#### Response

```json
{
  "status": "ok",
  "service": "didi-api",
  "version": "1.0.0"
}
```

### Code Search

**Endpoint:** `POST /api/search`

Perform semantic code search across the indexed repositories.

#### Request Body

```json
{
  "query": "websocket connection",  // (Required) The search query
  "limit": 5,                       // (Optional) Maximum results to return (default: 10)
  "detailed": false                 // (Optional) Whether to return detailed results (default: false)
}
```

#### Response

```json
{
  "query": "websocket connection",
  "count": 3,
  "results": [
    {
      "file_path": "src/hooks/useWebSocket.ts",
      "repo_name": "degenduel",
      "relevance": 0.923,
      "snippet": "const useWebSocket = (url: string) => {\n  const [socket, setSocket] = useState<WebSocket | null>(null);\n  // ... more code ... "
    },
    {
      "file_path": "src/contexts/WebSocketContext.tsx",
      "repo_name": "degenduel",
      "relevance": 0.887,
      "snippet": "// WebSocket context implementation..."
    },
    {
      "file_path": "docs/WEBSOCKET_UNIFIED_SYSTEM.md",
      "repo_name": "degenduel",
      "relevance": 0.764,
      "snippet": "# WebSocket Unified System\n\nThis document describes the architecture..."
    }
  ]
}
```

### Question Answering

**Endpoint:** `POST /api/ask`

Ask questions about the codebase and get AI-generated answers based on the code context.

#### Request Body

```json
{
  "question": "How does the websocket authentication work?",  // (Required) The question to ask
  "include_sources": true                                     // (Optional) Include source references (default: true)
}
```

#### Response

```json
{
  "answer": "WebSocket authentication in DegenDuel works through a multi-step process...",
  "sources": [
    {
      "file_path": "src/hooks/useAuth.ts",
      "repo_name": "degenduel",
      "relevance": 0.912,
      "content": "// Authentication implementation..."
    },
    {
      "file_path": "src/services/userService.ts",
      "repo_name": "degenduel_backend",
      "relevance": 0.876,
      "content": "// User service code..."
    }
  ]
}
```

### System Status

**Endpoint:** `GET /api/system/status`

Get information about Didi's system components.

#### Response

```json
{
  "database": {
    "exists": true,
    "size": "1.2G",
    "collection": "multi_codebase"
  },
  "repositories": {
    "total": 3,
    "active": 2,
    "names": ["DegenDuel", "DegenDuel Backend"]
  },
  "models": {
    "exists": true,
    "size": "4.5G"
  }
}
```

## Response Formats

All API responses follow a consistent format:

1. **Success Responses:**
   - HTTP Status Code: 200 OK
   - Content-Type: application/json
   - Body: JSON object with operation-specific data

2. **Error Responses:**
   - HTTP Status Code: Appropriate error code (400, 404, 500, etc.)
   - Content-Type: application/json
   - Body: `{ "error": "Error message" }`

## Error Handling

The API implements comprehensive error handling:

| HTTP Status | Description | Common Causes |
|-------------|-------------|--------------|
| 400 | Bad Request | Missing required parameters, invalid JSON |
| 404 | Not Found | Endpoint doesn't exist |
| 500 | Internal Server Error | Database issues, model loading failures |

Example error response:

```json
{
  "error": "Query parameter is required"
}
```

## Extending the API

Didi's API is designed to be easily extensible. Here's how to add a new endpoint:

### 1. Create a New Endpoint Function

Add a new route function to `api_server.py`:

```python
@app.route('/api/custom_endpoint', methods=['POST'])
def custom_endpoint():
    try:
        data = request.get_json()
        
        # Validate inputs
        if not data or 'required_param' not in data:
            return jsonify({"error": "Required parameter is missing"}), 400
        
        # Process the request using existing Didi functionality
        # ...
        
        # Return results
        return jsonify({
            "result": "Processed data",
            "additional_data": { ... }
        })
    
    except Exception as e:
        logger.error(f"Error in custom endpoint: {e}")
        return jsonify({"error": str(e)}), 500
```

### 2. Import Required Modules

Ensure any required modules or functions from Didi's codebase are properly imported:

```python
from scripts.some_module import some_function
```

### 3. Document the New Endpoint

Update this API_REFERENCE.md file to include documentation for your new endpoint.

### 4. Test the Endpoint

Test the new endpoint with curl or a REST client:

```bash
curl -X POST http://localhost:8000/api/custom_endpoint \
  -H "Content-Type: application/json" \
  -d '{"required_param": "value"}'
```

## Example Implementations

### Python Client

```python
import requests

class DidiClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def search(self, query, limit=10, detailed=False):
        payload = {
            "query": query,
            "limit": limit,
            "detailed": detailed
        }
        response = requests.post(f"{self.base_url}/api/search", json=payload)
        return response.json()
    
    def ask(self, question, include_sources=True):
        payload = {
            "question": question,
            "include_sources": include_sources
        }
        response = requests.post(f"{self.base_url}/api/ask", json=payload)
        return response.json()
    
    def system_status(self):
        response = requests.get(f"{self.base_url}/api/system/status")
        return response.json()

# Usage
client = DidiClient()
results = client.search("websocket implementation", limit=5)
answer = client.ask("How does user authentication work?")
```

### JavaScript Client

```javascript
class DidiClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async search(query, limit = 10, detailed = false) {
    const response = await fetch(`${this.baseUrl}/api/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        limit,
        detailed,
      }),
    });
    return response.json();
  }

  async ask(question, includeSources = true) {
    const response = await fetch(`${this.baseUrl}/api/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        include_sources: includeSources,
      }),
    });
    return response.json();
  }

  async systemStatus() {
    const response = await fetch(`${this.baseUrl}/api/system/status`);
    return response.json();
  }
}

// Usage
const client = new DidiClient();
client.search('websocket implementation', 5)
  .then(results => console.log(results));
client.ask('How does user authentication work?')
  .then(answer => console.log(answer));
```

### Command Line

```bash
# Health check
curl http://localhost:8000/health

# Search code
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "websocket connection", "limit": 5}'

# Ask a question
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does user authentication work?"}'

# System status
curl http://localhost:8000/api/system/status
```