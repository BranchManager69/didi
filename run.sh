#!/bin/bash
# Didi run script - starts Didi in Docker

# Check if environment is set up
if [ ! -d "/home/ubuntu/degenduel-gpu/models" ] || [ ! -d "/home/ubuntu/degenduel-gpu/repos" ]; then
    echo "Environment not set up, running startup script..."
    ./startup.sh
fi

# Build and run Docker container
docker-compose up --build