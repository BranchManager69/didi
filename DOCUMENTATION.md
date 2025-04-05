# Didi: DegenDuel's AI Assistant

## Complete Documentation

This document provides comprehensive documentation for Didi, a powerful RAG-based AI assistant designed for the DegenDuel codebase. Didi enables semantic code search, intelligent question answering, and codebase exploration across multiple repositories.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Persistent Storage Setup](#persistent-storage-setup)
3. [Installation and Setup](#installation-and-setup)
4. [Command Reference](#command-reference)
5. [Configuration](#configuration)
6. [Advanced Features](#advanced-features)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## System Architecture

Didi is built as a robust Retrieval-Augmented Generation (RAG) system specifically designed for code understanding. The system consists of several key components:

### Core Components

- **Vector Database**: ChromaDB stores embeddings for fast semantic search
- **Embedding Models**: HuggingFace embedding models transform code into semantic vectors
- **LLM Integration**: Uses CodeLlama for code-specific question answering
- **CLI Interface**: Streamlined bash interface for user interaction
- **Docker Support**: Containerization for consistent environment across instances

### Data Flow

1. **Indexing Phase**:
   - Code repositories are cloned/updated
   - Files are parsed and split into chunks
   - Embeddings are generated for each chunk
   - Vector data is stored in ChromaDB

2. **Query Phase**:
   - User queries are transformed into embeddings
   - Vector search retrieves relevant code chunks
   - LLM generates responses using the retrieved context
   - Results are presented with source information

---

## Persistent Storage Setup

Didi is designed to operate on ephemeral Lambda Labs instances with persistent storage. This architecture ensures data preservation across instance restarts.

### Directory Structure

```
/home/ubuntu/degenduel-gpu/         # Persistent storage mount point
│
├── models/                         # Local cache for ML models
│   ├── sentence-transformers/      # Embedding models
│   └── huggingface/                # LLM models
│
├── data/                           # Persistent data storage
│   └── chroma_db/                  # Vector database
│       ├── chroma.sqlite3          # Main database file
│       ├── repos_metadata.json     # Repository metadata
│       └── [collection directories]
│
├── repos/                          # Cloned code repositories
│   ├── degenduel/                  # Frontend repo
│   ├── degenduel_backend/          # Backend repo
│   └── [other repos]
│
└── config/                         # Configuration files
    └── repos_config.json           # Repository configuration
```

### Environment Variables

Didi uses the following environment variables to locate resources:

- `CODE_RAG_PATH`: Base directory for Didi
- `CODE_RAG_REPOS_PATH`: Directory containing code repositories
- `CODE_RAG_DB_PATH`: ChromaDB storage location
- `CODE_RAG_CONFIG_PATH`: Path to repository configuration
- `HF_HOME`: HuggingFace models cache
- `TORCH_HOME`: PyTorch models cache

---

## Installation and Setup

### Prerequisites

- Python 3.10+
- 16GB+ RAM (64GB+ recommended for large codebases)
- CUDA-compatible GPU (optional but recommended)
- Persistent storage volume

### One-Time Setup

```bash
# Clone Didi repository (if not already done)
git clone https://github.com/BranchManager69/didi.git
cd didi

# Run setup script
./didi.sh setup
```

The setup script:
1. Creates required directories in persistent storage
2. Installs Docker and Docker Compose (if not already installed)
3. Copies configuration to persistent storage
4. Installs Python dependencies
5. Copies repositories and database (if they exist)

### Building the Knowledge Base

```bash
# Build with parallel processing (recommended for large codebases)
./didi.sh parallel-index

# Or use standard indexing
./didi.sh index
```

The indexing process:
1. Loads documents from all enabled repositories
2. Processes files in parallel using multiple CPU cores
3. Generates embeddings for each document
4. Stores vector data in ChromaDB

Indexing time varies depending on:
- Repository size
- Number of CPU cores
- Available RAM
- Embedding model size

For a large codebase (>100,000 LOC), expect 10-30 minutes on first run.

---

## Command Reference

Didi provides a rich set of commands through its CLI interface:

### Search Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `./didi.sh get "query"` | `g`, `see` | Performs quick semantic code search |
| `./didi.sh details "query"` | `d` | Gets detailed code snippets for a query |
| `./didi.sh ask "question"` | `a` | Asks Didi a question about the code |
| `./didi.sh interactive` | `i` | Starts an interactive session with Didi |
| `./didi.sh test "query1" "query2"` | | Runs A/B tests on embedding models |

### Repository Management

| Command | Description |
|---------|-------------|
| `./didi.sh add-repo "name" path/url [description]` | Adds a new repository to Didi |
| `./didi.sh list-repos` | Lists all configured repositories |
| `./didi.sh enable-repo repo_key` | Enables a repository |
| `./didi.sh disable-repo repo_key` | Disables a repository |
| `./didi.sh update` | Updates all repositories and rebuilds if needed |

### System Commands

| Command | Description |
|---------|-------------|
| `./didi.sh setup` | Sets up Didi's environment |
| `./didi.sh status` | Checks Didi's system status |
| `./didi.sh index` | Forces rebuild of knowledge base |
| `./didi.sh parallel-index` | Forces rebuild with parallel processing |
| `./didi.sh docker` | Runs Didi in a Docker container |
| `./didi.sh help` | Shows help message |

---

## Configuration

### Repository Configuration

Didi's repositories are configured in `/home/ubuntu/degenduel-gpu/config/repos_config.json`:

```json
{
  "repo_key": {
    "name": "Repository Name",
    "description": "Repository description",
    "path": "/path/to/repository",
    "git_url": "https://github.com/user/repo.git",
    "enabled": true
  },
  ...
}
```

Each repository entry contains:
- `repo_key`: Unique identifier for the repository
- `name`: Display name for the repository
- `description`: Description of the repository
- `path`: Absolute path to the repository
- `git_url`: URL for cloning/updating (optional)
- `enabled`: Whether to include the repository in indexing

### Embedding Models

Didi supports multiple embedding models for different use cases. These are configured in `config.py`:

```python
EMBEDDING_MODELS = {
    # Original model - general purpose
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    
    # Code-specific model - better for code understanding
    "code": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    
    # Larger, more powerful general model
    "mpnet": "sentence-transformers/all-mpnet-base-v2"
}
```

### A/B Testing Configuration

Didi supports A/B testing of embedding models through configuration in `config.py`:

```python
# A/B testing configuration
# Set to True to enable A/B testing of embedding models
ENABLE_AB_TESTING = os.environ.get("CODE_RAG_AB_TESTING", "False").lower() == "true"

# If A/B testing is enabled, use this as the second model
AB_TEST_MODEL = os.environ.get("CODE_RAG_AB_TEST_MODEL", EMBEDDING_MODELS["code"])

# A/B testing metrics collection directory
METRICS_DIR = BASE_DIR / "metrics"
```

---

## Advanced Features

### Parallel Processing

Didi uses parallel processing to dramatically speed up indexing:

- **Multi-Process File Loading**: Processes repositories in parallel
- **Multi-Thread Document Processing**: Handles file I/O concurrently
- **Optimized Embedding Generation**: Batches embeddings for efficiency

To utilize this feature, use:
```bash
./didi.sh parallel-index
```

The system automatically detects the optimal number of workers based on available CPU cores.

### Hybrid Search

Didi implements hybrid search combining:
- **Vector Search**: For semantic understanding
- **BM25 Search**: For keyword relevance

This approach provides better search results by:
1. Using semantic search to understand concepts
2. Using keyword search to enforce term presence
3. Combining results using reciprocal rank fusion

### Interactive Mode

The interactive mode provides a continuous session with Didi:

```bash
./didi.sh interactive
```

This mode:
- Maintains context between queries
- Provides more detailed responses
- Offers improved source information
- Eliminates the need to re-type the command prefix

### Docker Support

Didi can run in a Docker container for consistent environment:

```bash
./didi.sh docker
```

The Docker setup:
- Mounts persistent storage volumes
- Sets up the correct environment variables
- Ensures consistent dependencies
- Isolates the application environment

---

## Performance Optimization

### System Requirements

Didi's performance is highly dependent on system resources:

| Resource | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| CPU Cores | 4 | 16 | 64+ |
| RAM | 8GB | 16GB | 64GB+ |
| Storage | 10GB | 50GB | 100GB+ |
| GPU | None | CUDA-compatible | A100/H100 |

### Memory Optimization

Didi implements several memory optimization techniques:

1. **Model Quantization**: Uses 8-bit quantization for LLMs
2. **Chunked Processing**: Processes large repositories in chunks
3. **Batched Operations**: Combines operations to reduce overhead
4. **Caching**: Uses persistent caching for models and embeddings

### Indexing Speed

Indexing large codebases can be optimized by:

1. Increasing the `chunk_size` parameter in the index script
2. Adjusting the number of parallel workers
3. Using a smaller embedding model for faster processing
4. Disabling unnecessary repositories

### Query Performance

For optimal query performance:

1. Use specific queries rather than general ones
2. Adjust the `TOP_K` parameter for more/fewer results
3. Use interactive mode to avoid startup overhead
4. Run on a GPU-equipped instance for fastest LLM processing

---

## Troubleshooting

### Common Issues

#### Indexing Errors

**Problem**: ChromaDB errors during indexing
**Solution**: 
1. Check directory permissions on persistent storage
2. Delete the ChromaDB directory and try again
3. Verify enough disk space is available

**Problem**: Out of memory during indexing
**Solution**:
1. Reduce the number of worker processes
2. Process repositories one at a time
3. Use a smaller embedding model

#### Query Errors

**Problem**: LLM fails to load
**Solution**:
1. Check if model files are correctly downloaded
2. Verify enough GPU/RAM is available
3. Try a smaller model by adjusting `DEFAULT_MODEL_PATH`

**Problem**: No results from search
**Solution**:
1. Check if repositories are correctly indexed
2. Try broader search terms
3. Verify the ChromaDB database exists and has data

### Logs and Debugging

Didi logs information to the console during operation. For detailed logs:

```bash
# Run with more verbose logging
PYTHONPATH=$PYTHONPATH:/home/ubuntu/didi python -m scripts.enhanced_query -i --debug
```

Key log files:
- ChromaDB logs: Found in the ChromaDB directory
- Python module logs: Standard output

### Reinstalling

If you encounter persistent issues, you can completely reset Didi:

```bash
# Remove the ChromaDB directory
rm -rf /home/ubuntu/degenduel-gpu/data/chroma_db

# Reinstall Python dependencies
pip install -r requirements.txt

# Rebuild the index
./didi.sh parallel-index
```

This will:
1. Remove the existing vector database
2. Reinstall all required dependencies
3. Rebuild the knowledge base from scratch

---

## Final Notes

Didi is designed to be a powerful assistant for navigating and understanding the DegenDuel codebase. By using advanced RAG techniques, parallel processing, and persistent storage, it provides a robust solution for code search and comprehension.

For updates and improvements, refer to the repository at:
https://github.com/BranchManager69/didi

---

Documentation prepared for DegenDuel by Didi on April 5, 2025