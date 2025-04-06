#!/bin/bash
# Non-blocking script to run Didi API server in the background

# Create a log file directory
LOG_DIR="/home/ubuntu/degenduel-gpu/repos/didi/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/didi_api.log"

# Kill any existing API server process
pkill -f "python.*api_server.py" > /dev/null 2>&1 || true

# Default port
PORT=8000
if [ ! -z "$1" ]; then
  PORT="$1"
fi

# Start server completely detached
cd /home/ubuntu/degenduel-gpu/repos/didi
nohup python scripts/api_server.py --port "$PORT" --use-gunicorn --workers 4 > "$LOG_FILE" 2>&1 &

# Get PID and record it
PID=$!
echo "$PID" > /home/ubuntu/degenduel-gpu/repos/didi/api_server.pid
echo "Didi API server started on port $PORT with PID $PID"
echo "Logs available at: $LOG_FILE"
echo "To stop the server: kill $PID"