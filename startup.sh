#!/bin/bash
# Didi startup script - ensures proper setup when the instance restarts

# Set up environment variables
export HF_HOME=/home/ubuntu/degenduel-gpu/models
export TORCH_HOME=/home/ubuntu/degenduel-gpu/models
export CODE_RAG_PATH=/home/ubuntu/didi
export CODE_RAG_REPOS_PATH=/home/ubuntu/degenduel-gpu/repos
export CODE_RAG_DB_PATH=/home/ubuntu/degenduel-gpu/data/chroma_db
export CODE_RAG_CONFIG_PATH=/home/ubuntu/degenduel-gpu/config/repos_config.json

# Create required directories if they don't exist
mkdir -p $HF_HOME
mkdir -p $CODE_RAG_DB_PATH
mkdir -p $CODE_RAG_REPOS_PATH
mkdir -p $(dirname $CODE_RAG_CONFIG_PATH)

# Check if repos_config.json exists in persistent storage
if [ ! -f "$CODE_RAG_CONFIG_PATH" ]; then
    echo "repos_config.json not found in persistent storage, copying from local..."
    cp /home/ubuntu/didi/repos_config.json $CODE_RAG_CONFIG_PATH
fi

# Check if repositories exist in persistent storage
# If not, copy them from local or clone from git
for repo in $(ls -1 /home/ubuntu/didi/repos); do
    if [ ! -d "$CODE_RAG_REPOS_PATH/$repo" ]; then
        echo "Repository $repo not found in persistent storage, copying from local..."
        mkdir -p $CODE_RAG_REPOS_PATH/$repo
        cp -r /home/ubuntu/didi/repos/$repo/* $CODE_RAG_REPOS_PATH/$repo/
    fi
done

# Check if ChromaDB exists in persistent storage
if [ ! -d "$CODE_RAG_DB_PATH" ] || [ -z "$(ls -A $CODE_RAG_DB_PATH)" ]; then
    echo "ChromaDB not found in persistent storage, copying from local..."
    mkdir -p $CODE_RAG_DB_PATH
    cp -r /home/ubuntu/didi/chroma_db/* $CODE_RAG_DB_PATH/
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found, installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose not found, installing..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.6/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

echo "Didi environment is ready! You can now run Didi using:"
echo "cd /home/ubuntu/didi && ./run.sh"