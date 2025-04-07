# Didi Performance Guide for GH200

This guide provides information on optimizing Didi for maximum performance on the Lambda Labs GH200 (Grace Hopper) GPU with 480GB VRAM.

## GH200 Optimization

The GH200 (Grace Hopper) GPU provides 480GB of VRAM, which allows us to run the largest language models without quantization for maximum quality. Didi has been specially configured to take advantage of this hardware:

### Ultra Profile

An `ultra` profile has been created specifically for the GH200, which includes:

- Llama-3-70B-Instruct as the default model
- Largest possible context window (65536 tokens)
- Maximum new tokens for generation (8192)
- Flash Attention 2 for faster inference
- BFloat16 precision for optimal performance
- Higher GPU memory utilization (95%)

The profile configuration is in `model_profiles/ultra.json`:

```json
{
  "name": "UltraRAG (GH200 Optimized)",
  "llm_model": "meta-llama/Meta-Llama-3-70B-Instruct",
  "embed_model": "intfloat/e5-large-v2",
  "collection_name": "degenduel_code_ultra",
  "chunk_size": 32768,
  "chunk_overlap": 4096,
  "context_window": 65536,
  "max_new_tokens": 8192,
  "temperature": 0.1,
  "gpu_memory_utilization": 0.95,
  "load_in_8bit": false,
  "load_in_4bit": false,
  "use_flash_attn": true
}
```

### Automatic Hardware Detection

Didi's API server includes automatic hardware detection that will:

1. Check for CUDA GPU availability
2. Detect the GPU model and available VRAM
3. Automatically select the appropriate profile based on hardware capabilities:
   - For GH200 (detected by name or VRAM > 400GB): `ultra` profile
   - For high-end GPUs (VRAM > 60GB): `gh200` profile
   - For powerful GPUs (VRAM > 30GB): `powerful` profile
   - For good consumer GPUs (VRAM > 15GB): `llama3` profile
   - For mid-range GPUs (VRAM > 8GB): `mistral` profile
   - For limited GPUs or CPU-only: `phi3` profile

### Manual Profile Selection

You can manually override the profile selection using the `DIDI_MODEL_PROFILE` environment variable:

```bash
DIDI_MODEL_PROFILE=ultra python scripts/api_server.py
```

Or force the GH200 profile regardless of environment variable:

```bash
DIDI_FORCE_GH200=true python scripts/api_server.py
```

## Performance Tuning

### Transformer Engine

On GH200, you can enable NVIDIA's Transformer Engine for additional performance:

```bash
USE_TRANSFORMER_ENGINE=1 python scripts/api_server.py
```

This enables specialized NVIDIA optimizations for transformer models.

### Parallel Processing

For indexing large codebases, use the parallel indexing script:

```bash
python scripts/parallel_index.py
```

This utilizes multiple CPU cores for faster document processing and embedding generation.

### Memory Management

To control GPU memory usage, adjust the `gpu_memory_utilization` parameter in your profile:

- Higher values (0.95) maximize performance but may cause OOM errors with other applications
- Lower values (0.7) are safer but may reduce performance slightly

## Benchmarking

You can benchmark Didi's performance on your hardware using:

```bash
python scripts/benchmark.py
```

This will test various operations and report timings for:
- Model loading time
- Embedding generation time
- Query processing time
- Response generation time

## Optimizing for Production

For production deployments on the GH200:

1. Use Gunicorn with multiple workers:
   ```bash
   python scripts/api_server.py --use-gunicorn --workers 4
   ```

2. Enable result caching to improve response times for repeated queries:
   ```bash
   ENABLE_RESULT_CACHE=true python scripts/api_server.py
   ```

3. Use the optimized `run_api.sh` script which includes all optimizations:
   ```bash
   ./run_api.sh
   ```

## Profile Comparison

| Profile | Model | VRAM Required | Context Window | Good For |
|---------|-------|---------------|----------------|----------|
| ultra   | Llama-3-70B | 400GB+ | 65536 | GH200 GPU |
| gh200   | Llama-3-70B | 80GB+ | 32768 | A100 80GB |
| powerful | CodeLlama-34B | 40GB+ | 8192 | A10G, A100 40GB |
| llama3 | Llama-3-8B | 16GB+ | 8192 | Consumer GPUs (4090) |
| mistral | Mistral-7B | 8GB+ | 8192 | Mid-range GPUs |
| phi3 | Phi-3-mini | 4GB+ | 4096 | Limited GPUs or CPU |

## Additional Optimizations

### Advanced Embedding Models

For best results on the GH200, we use the `intfloat/e5-large-v2` embedding model which provides:
- Better semantic understanding of code
- More accurate search results
- Improved context retention

### Optimized Chunking Strategy

The ultra profile uses:
- Larger chunk sizes (32768 characters)
- More overlap between chunks (4096 characters)
- This enables better understanding of large code files and complex relationships

### Hybrid Search Implementation

Didi combines:
- Vector search for semantic understanding
- BM25/keyword search for exact matches
- Result re-ranking to prioritize most relevant code

### Automatic API Server Scaling

The API server automatically:
1. Detects the GH200 hardware
2. Configures optimal worker count
3. Sets memory limits for maximum performance
4. Enables FlashAttention 2 for 2-3x faster inference

## Lambda Labs GH200 Environment Notes

To maintain optimal performance:

1. **Environment Variables**:
   - `DIDI_MODEL_PROFILE=ultra` (set in run_api.sh)
   - `TRANSFORMERS_CACHE=/home/ubuntu/degenduel-gpu/models` (for persistent model storage)

2. **Autostart Configuration**:
   - The systemd service is configured to start after the persistent storage mounts
   - Memory limits are set appropriately for the GH200

3. **Storage Management**:
   - Keep the persistent storage (`/home/ubuntu/degenduel-gpu/`) under 3TB
   - Run regular maintenance to clean old logs and cached models

4. **API Usage**:
   - The HTTP API is optimized for the GH200
   - For best performance, use batched requests
   - The API automatically detects and uses the ultra profile