# Didi with Ollama and Llama 4

This integration allows you to use Didi with Meta's Llama 4 model running locally through Ollama. This provides a high-quality open source LLM alternative to the default HuggingFace models.

## Setup

1. Install Ollama:
   ```bash
   bash setup_ollama.sh
   ```

   This script will:
   - Install Ollama if not already installed
   - Start the Ollama service if not running
   - Download the Llama 4 model (approximately 40GB)

2. Once the setup is complete, you can use Didi with Llama 4:
   ```bash
   ./didi_ollama.sh interactive
   ```

## Usage

The `didi_ollama.sh` script is designed to be a drop-in replacement for the regular `didi.sh`. It accepts all the same commands:

```bash
# Start an interactive session with Llama 4
./didi_ollama.sh interactive

# Ask a specific question
./didi_ollama.sh ask "How does the Solana integration work?"

# Search for code
./didi_ollama.sh get "websocket authentication"

# Get detailed code snippets
./didi_ollama.sh details "token price calculation"

# Check system status
./didi_ollama.sh status
```

## How It Works

The integration uses a custom `OllamaLLM` class that implements the same interface that `llama-index` expects, but communicates with the Ollama service via its REST API. This allows for seamless substitution of Llama 4 for HuggingFace models while maintaining all of Didi's retrieval and search functionality.

The integration automatically sets the following environment variables:

- `DIDI_USE_OLLAMA=true` - Tells Didi to use Ollama instead of HuggingFace
- `OLLAMA_MODEL=llama4` - Specifies which model to use
- `OLLAMA_URL=http://localhost:11434` - Specifies the Ollama service URL

## Advanced Configuration

You can modify these settings by editing the `didi_ollama.sh` script.

### Changing the Model

To use a different model, modify the `OLLAMA_MODEL` environment variable in `didi_ollama.sh`. Other models available through Ollama include:

- `llama4:latest` - The latest Llama 4 model
- `llama3` - Meta's Llama 3
- `mistral` - Mistral AI's model
- `gemma` - Google's Gemma model

View all available models with `ollama list` or pull new ones with `ollama pull <model>`.

### Performance Considerations

Llama 4 is a large model and requires significant GPU resources. For optimal performance:

- Ensure you have at least 40GB of free VRAM
- The model runs best on a dedicated GPU

## Troubleshooting

If you encounter issues:

1. Check if Ollama service is running: `ps aux | grep ollama`
2. Verify Llama 4 is downloaded: `ollama list | grep llama4`
3. Check Ollama logs: `journalctl -u ollama`
4. Restart Ollama service: `sudo systemctl restart ollama`
5. Verify API connectivity: `curl http://localhost:11434/api/version`

## Reverting to HuggingFace Model

If you need to return to using the HuggingFace model:

```bash
# Use the standard didi script
./didi.sh interactive
```