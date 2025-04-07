#!/bin/bash
# Auto-restart script for Didi API when UI files change

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

# Configuration
DIDI_ROOT="/home/ubuntu/degenduel-gpu/didi"
WATCH_DIRS="$DIDI_ROOT/public $DIDI_ROOT/scripts $DIDI_ROOT/model_profiles"
LOG_FILE="$DIDI_ROOT/logs/auto-restart.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

log() {
  local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
  echo -e "$timestamp - $1" | tee -a "$LOG_FILE"
}

restart_server() {
  log "${YELLOW}Change detected in $1. Restarting server...${RESET}"
  
  # Kill existing server process
  pkill -f "api_server.py" || true
  
  # Wait for process to fully terminate
  sleep 2
  
  # Start server in background
  cd "$DIDI_ROOT"
  ./run_api.sh 8000 &
  
  # Wait for server to start
  sleep 3
  
  # Check if server is running
  if pgrep -f "api_server.py" > /dev/null; then
    log "${GREEN}Server successfully restarted!${RESET}"
  else
    log "${YELLOW}Server may have failed to restart. Check logs.${RESET}"
  fi
}

# Install inotify-tools if not present
if ! command -v inotifywait &> /dev/null; then
  log "${BLUE}Installing inotify-tools...${RESET}"
  sudo apt-get update && sudo apt-get install -y inotify-tools
fi

log "${GREEN}Starting auto-restart monitor for Didi API${RESET}"
log "${BLUE}Watching directories: $WATCH_DIRS${RESET}"
log "${BLUE}Press Ctrl+C to stop monitoring${RESET}"

# First start the server if it's not running
if ! pgrep -f "api_server.py" > /dev/null; then
  log "${YELLOW}Server not running. Starting it now...${RESET}"
  cd "$DIDI_ROOT"
  ./run_api.sh 8000 &
  sleep 3
fi

# Monitor for changes
while true; do
  inotifywait -r -e modify,create,delete,move $WATCH_DIRS | while read dir event file; do
    # Only restart for relevant file types
    if [[ "$file" =~ \.(html|css|js|json|py)$ ]]; then
      restart_server "$dir/$file"
    fi
  done
done