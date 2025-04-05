# Didi Performance Guide

This guide provides advanced optimization techniques for running Didi with maximum performance, especially when working with large codebases.

## Hardware Optimization

### CPU Utilization

Didi's indexing performance scales with CPU cores. The parallel indexing implementation automatically detects your system's capabilities.

```bash
# Run with optimal parallel settings
./didi.sh parallel-index
```

On your 64-core system, this will:
- Use 32-60 worker threads for document processing
- Process multiple repositories concurrently
- Reserve some cores for system operations

### Memory Management

With 525GB of RAM available, you can optimize for maximum performance:

```python
# Add to config.py for larger chunk sizes
CHUNK_SIZE = 2048       # Increased from default 1024
CHUNK_OVERLAP = 256     # Increased from default 128
```

These settings allow for:
- Processing larger code blocks as single units
- Better context preservation between chunks
- More comprehensive search results

### Storage Configuration

Your 4TB disk allows for enhanced caching:

```python
# Add to config.py for more aggressive caching
ENABLE_EMBEDDING_CACHE = True
CACHE_FOLDER = "/home/ubuntu/degenduel-gpu/cache"
```

This configuration:
- Caches embedding results for faster reindexing
- Stores model weights for quick loading
- Preserves data between sessions

## Model Selection

### Embedding Models

For your high-performance environment, consider using larger embedding models:

```python
# Update in config.py
DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Larger, more powerful model
```

Alternative high-performance models:
- `BAAI/bge-large-en-v1.5` - State-of-the-art retrieval
- `text-embedding-3-large` if using OpenAI API
- `intfloat/e5-large-v2` - Excellent code understanding

### LLM Selection

With your hardware, you can use larger models for better answers:

```python
# Update in config.py
DEFAULT_MODEL_PATH = "codellama/CodeLlama-13b-instruct-hf"  # Larger model
```

Other options:
- `codellama/CodeLlama-34b-instruct-hf` for highest quality
- `mistralai/Mistral-7B-Instruct-v0.2` for balanced performance
- `bigcode/starcoder2-15b` specialized for code

## Indexing Optimizations

### Filter Irrelevant Files

Adjust ignored directories and file patterns to focus on relevant code:

```python
# Update in config.py
IGNORE_DIRS = [
    ".git", 
    "node_modules", 
    "dist", 
    "dist-dev", 
    ".next", 
    "coverage", 
    "out",
    "public/assets",     # Add large asset directories
    "tests/fixtures",    # Add test fixtures
]

# Add file size limits
MAX_FILE_SIZE_MB = 5    # Skip files larger than 5MB
```

### Code Parsing Improvements

Implement specialized code parsing:

```python
# Advanced code parsing (add to config.py)
USE_TREE_SITTER = True  # Use tree-sitter for code parsing
PARSE_DOCSTRINGS = True # Extract docstrings for better context
```

This enables:
- Language-aware code chunking
- Function/class level segmentation
- Better preservation of code structures

## Query Optimizations

### Hybrid Search

For most accurate results, use hybrid search combining:
- Vector search for semantic understanding
- BM25/keyword search for exact matches

```python
# Already implemented in enhanced_query.py
# Parameters to tune:
TOP_K = 15              # Number of results to retrieve initially
RERANK_TOP_N = 10       # Number to keep after reranking
```

### Response Generation

Tune response generation for deeper code understanding:

```python
# Parameters in enhanced_query.py
MAX_NEW_TOKENS = 2048   # Increased from 1024 for more detailed answers
TEMPERATURE = 0.1       # Keep low for deterministic answers
```

## Monitoring and Tuning

### Performance Metrics

Monitor key metrics to identify bottlenecks:
- Indexing time per repository
- Embedding generation time
- Query latency
- RAM usage during operation

The A/B testing feature can help compare different configurations:

```bash
./didi.sh test "websocket implementation" "user authentication" "contest creation"
```

### Regular Maintenance

For optimal performance:

1. Run updates regularly to capture new code changes:
   ```bash
   ./didi.sh update
   ```

2. Periodically clean and rebuild the index:
   ```bash
   rm -rf /home/ubuntu/degenduel-gpu/data/chroma_db/*
   ./didi.sh parallel-index
   ```

3. Monitor disk usage on persistent storage:
   ```bash
   du -sh /home/ubuntu/degenduel-gpu/*
   ```

## Docker Optimization

When running in Docker, tune container resources:

```yaml
# In docker-compose.yml
services:
  didi:
    # ...
    deploy:
      resources:
        limits:
          cpus: '60'    # Reserve some CPUs for system
          memory: 500G  # Almost all available RAM
```

## LambdaLabs-Specific Optimizations

For optimal performance on LambdaLabs instances:

1. **Instance Persistence**: 
   - Store all important data in `/home/ubuntu/degenduel-gpu/`
   - Set environment variables in `.bashrc` for quick restart

2. **GPU Acceleration**:
   - Enable GPU acceleration for LLM inference
   - Set `device_map: "auto"` in model configuration (already implemented)

3. **Instance Type Selection**:
   - Use instances with local NVMe storage for fastest disk I/O
   - Select highest CPU count for parallel processing

By implementing these optimizations, Didi will provide:
- Faster indexing of large codebases
- More accurate code search results
- More detailed and helpful answers
- Better user experience overall

---

For additional performance tuning, consult the full technical documentation.