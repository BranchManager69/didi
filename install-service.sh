#!/bin/bash
# Install Didi API as a systemd service for auto-start

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up Didi API auto-start service...${NC}"

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run with sudo:${NC} sudo ./install-service.sh"
  exit 1
fi

# Ensure the service file exists
if [ ! -f "didi-api.service" ]; then
  echo -e "${RED}Service file not found!${NC}"
  exit 1
fi

# Copy service file to systemd directory
cp didi-api.service /etc/systemd/system/

# Reload systemd daemon
systemctl daemon-reload

# Enable and start the service
systemctl enable didi-api.service
systemctl start didi-api.service

# Check status
sleep 2
status=$(systemctl is-active didi-api.service)

if [ "$status" = "active" ]; then
  echo -e "${GREEN}Didi API service installed and started successfully!${NC}"
  echo -e "${GREEN}Service will automatically start on system boot.${NC}"
  echo -e "\n${YELLOW}To check status:${NC} sudo systemctl status didi-api.service"
  echo -e "${YELLOW}To stop service:${NC} sudo systemctl stop didi-api.service"
  echo -e "${YELLOW}To disable autostart:${NC} sudo systemctl disable didi-api.service"
  
  # Show the service status
  systemctl status didi-api.service --no-pager
else
  echo -e "${RED}Service installation failed. Please check the logs:${NC}"
  echo -e "journalctl -u didi-api.service"
fi