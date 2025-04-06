#!/bin/bash
# Setup script for Ollama with Llama 4 model
# -----------------------------------------------------

set -e

echo "Setting up Ollama with Llama 4 model"
echo "----------------------------------------------------"

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed successfully"
fi

# Check if Ollama service is running
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama service is already running"
else
    echo "Starting Ollama service..."
    ollama serve &
    # Wait for Ollama to start
    sleep 5
    echo "✅ Ollama service started"
fi

# Check if Llama 4 model is already downloaded
if ollama list | grep -q "llama4"; then
    echo "✅ Llama 4 model is already downloaded"
else
    echo "Downloading Llama 4 model..."
    echo "This may take a while depending on your connection speed"
    echo "Llama 4 is approximately 40GB in size"
    ollama pull llama4
    echo "✅ Llama 4 model downloaded successfully"
fi

echo "Ollama setup complete!"
echo "You can now use Didi with Llama 4 using ./didi_ollama.sh"