#!/bin/bash
# Script to run Didi API server

# Set up environment variables
export CODE_RAG_PATH="/home/ubuntu/degenduel-gpu/didi"
export CODE_RAG_REPOS_PATH="/home/ubuntu/degenduel-gpu/repos"
export CODE_RAG_DB_PATH="/home/ubuntu/degenduel-gpu/data/chroma_db"
export CODE_RAG_CONFIG_PATH="/home/ubuntu/degenduel-gpu/config/repos_config.json"
export HF_HOME="/home/ubuntu/degenduel-gpu/models"
export TORCH_HOME="/home/ubuntu/degenduel-gpu/models"
export DIDI_MODEL_PROFILE="ultra"  # Always use the best model for GH200

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

# Run directly (not detached) for systemd service
exec python scripts/api_server.py --port "$PORT" --use-gunicorn --workers 4 >> "$LOG_FILE" 2>&1