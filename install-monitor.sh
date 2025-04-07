#!/bin/bash
# Install Didi API auto-restart monitor as a systemd service

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up Didi API auto-restart monitor service...${NC}"

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run with sudo:${NC} sudo ./install-monitor.sh"
  exit 1
fi

# Ensure the service file exists
if [ ! -f "didi-api-monitor.service" ]; then
  echo -e "${RED}Service file not found!${NC}"
  exit 1
fi

# Install inotify-tools if not present
if ! command -v inotifywait &> /dev/null; then
  echo -e "${YELLOW}Installing inotify-tools...${NC}"
  apt-get update && apt-get install -y inotify-tools
fi

# Make script executable
chmod +x auto-restart.sh

# Copy service file to systemd directory
cp didi-api-monitor.service /etc/systemd/system/

# Reload systemd daemon
systemctl daemon-reload

# Enable and start the service
systemctl enable didi-api-monitor.service
systemctl start didi-api-monitor.service

# Check status
sleep 2
status=$(systemctl is-active didi-api-monitor.service)

if [ "$status" = "active" ]; then
  echo -e "${GREEN}Didi API monitor service installed and started successfully!${NC}"
  echo -e "${GREEN}Service will automatically start on system boot.${NC}"
  echo -e "\n${YELLOW}To check status:${NC} sudo systemctl status didi-api-monitor.service"
  echo -e "${YELLOW}To stop service:${NC} sudo systemctl stop didi-api-monitor.service"
  echo -e "${YELLOW}To disable autostart:${NC} sudo systemctl disable didi-api-monitor.service"
  
  # Show the service status
  systemctl status didi-api-monitor.service --no-pager
else
  echo -e "${RED}Service installation failed. Please check the logs:${NC}"
  echo -e "journalctl -u didi-api-monitor.service"
fi