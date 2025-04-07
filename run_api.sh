#!/bin/bash
# Script to run Didi API server optimized for GH200 GPU

# Set up environment variables
export CODE_RAG_PATH="/home/ubuntu/degenduel-gpu/didi"
export CODE_RAG_REPOS_PATH="/home/ubuntu/degenduel-gpu/repos"
export CODE_RAG_DB_PATH="/home/ubuntu/degenduel-gpu/data/chroma_db"
export CODE_RAG_CONFIG_PATH="/home/ubuntu/degenduel-gpu/config/repos_config.json"
export HF_HOME="/home/ubuntu/degenduel-gpu/models"
export TORCH_HOME="/home/ubuntu/degenduel-gpu/models"
export TRANSFORMERS_CACHE="/home/ubuntu/degenduel-gpu/models"

# GH200 optimization settings
export DIDI_MODEL_PROFILE="ultra"  # Use ultra-optimized profile for GH200
export USE_FLASH_ATTENTION=1       # Enable FlashAttention for faster inference
export TOKENIZERS_PARALLELISM=true # Parallelize tokenization
export USE_TF32=1                  # Enable TF32 for faster computation
export USE_TRANSFORMER_ENGINE=1    # Enable NVIDIA Transformer Engine optimizations

# Log file for output
LOG_DIR="/home/ubuntu/degenduel-gpu/didi/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/didi_api.log"

# Default port
PORT=8000
if [ ! -z "$1" ]; then
  PORT="$1"
fi

# Add timestamp to log
echo "Starting Didi API server on $(date)" >> "$LOG_FILE"

# Check for required directories
mkdir -p /home/ubuntu/degenduel-gpu/models
mkdir -p /home/ubuntu/degenduel-gpu/data/chroma_db

# Change to the script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
  source venv/bin/activate
fi

# Install required packages if needed
if ! pip list | grep -q flask; then
  echo "Installing required packages..." | tee -a "$LOG_FILE"
  pip install flask flask-cors gunicorn >> "$LOG_FILE" 2>&1
fi

# Start the server with gunicorn for production use
echo "Starting Didi API server on port $PORT" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"

# Print GH200 optimization information
echo "Running with GH200 optimizations:" | tee -a "$LOG_FILE"
echo " - Ultra profile (Llama-3-70B model)" | tee -a "$LOG_FILE"
echo " - FlashAttention 2 enabled" | tee -a "$LOG_FILE"
echo " - NVIDIA Transformer Engine enabled" | tee -a "$LOG_FILE"
echo " - TF32 computation enabled" | tee -a "$LOG_FILE"
echo " - 4 Gunicorn workers" | tee -a "$LOG_FILE"

# Run directly (not detached) for systemd service
exec python scripts/api_server.py --port "$PORT" --use-gunicorn --workers 4 >> "$LOG_FILE" 2>&1