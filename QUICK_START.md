# Didi - Quick Start Guide

## What is Didi?

Didi is DegenDuel's AI Assistant, a powerful tool designed to help you search, understand, and navigate the DegenDuel codebase. Using advanced Retrieval-Augmented Generation (RAG), Didi can answer questions about code implementation, find relevant files, and provide contextual information about the entire codebase.

## Setup (One-Time)

```bash
# Navigate to the Didi directory
cd /home/ubuntu/didi

# Run the setup script
./didi.sh setup
```

This setup script:
- Creates the necessary directory structure in persistent storage
- Installs required dependencies
- Configures the environment for optimal performance

## Building the Knowledge Base

```bash
# Build using parallel processing (recommended)
./didi.sh parallel-index
```

This process will:
- Load all code from configured repositories
- Generate embeddings for semantic search
- Store the knowledge base in persistent storage
- This may take 10-30 minutes for a large codebase

## Using Didi

### Interactive Mode (Recommended)

Start an interactive session with Didi:

```bash
./didi.sh interactive
```

In this mode, you can ask questions directly without repeating the command prefix.

### Ask Questions

Ask Didi about the codebase:

```bash
./didi.sh ask "How is websocket authentication implemented?"
```

### Search for Code

Find relevant code snippets:

```bash
./didi.sh get "user authentication flow"
```

Get detailed code snippets:

```bash
./didi.sh details "contest creation process"
```

## Repository Management

Add a new repository:

```bash
./didi.sh add-repo "NewRepo" /path/to/repo
```

List all repositories:

```bash
./didi.sh list-repos
```

Enable/disable a repository:

```bash
./didi.sh enable-repo repo_key
./didi.sh disable-repo repo_key
```

Update all repositories:

```bash
./didi.sh update
```

## System Commands

Check system status:

```bash
./didi.sh status
```

View help:

```bash
./didi.sh help
```

Run Didi in Docker:

```bash
./didi.sh docker
```

## Persistent Storage

Didi uses persistent storage to ensure data is preserved across Lambda Labs instance restarts:

```
/home/ubuntu/degenduel-gpu/
├── models/         # ML model cache
├── data/           # Vector database
├── repos/          # Code repositories
└── config/         # Configuration files
```

## For More Information

See the full documentation in `DOCUMENTATION.md` for detailed information on:
- Advanced features
- Configuration options
- Performance optimization
- Troubleshooting

---

**Note**: This quick start guide assumes you're running on a Lambda Labs instance with persistent storage mounted at `/home/ubuntu/degenduel-gpu/`.