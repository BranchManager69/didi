# Didi: DegenDuel's AI Assistant

A powerful AI assistant that helps you navigate and understand the DegenDuel codebase.

![Didi Demo](https://i.ibb.co/f2smYQH/coderag-diagram.png)

## 📋 Overview

Didi transforms the DegenDuel codebase into a searchable knowledge base, enabling:

- 🔍 **Semantic Code Search**: Find relevant code based on meaning, not just keywords
- 📝 **Detailed Code Exploration**: Get full context and implementation details
- 🤖 **AI-Powered Insights**: Ask Didi complex questions about the DegenDuel codebase
- 🚀 **Simple Command Interface**: Easy-to-use shell commands for all functionality

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Git
- DegenDuel repository

### Setup

```bash
# Clone this repository
git clone https://github.com/degenduel/didi.git
cd didi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Clone the DegenDuel repository if needed
git clone https://github.com/degenduel/degenduel-fe.git repo

# Build Didi's knowledge base
./didi.sh index
```

## 🚀 Usage

Didi provides a simple command-line interface for all functionality:

```bash
# Show available commands
./didi.sh help

# Check Didi's status
./didi.sh status

# Search code
./didi.sh search "websocket authentication"

# Detailed code view
./didi.sh details "contest system implementation"

# Ask Didi questions about the code
./didi.sh ask "How does the user authentication work?"

# Update repository and rebuild knowledge base
./didi.sh update
```

## 🔧 Configuration

Didi can be configured via environment variables or by editing `config.py`:

```bash
# Set custom paths via environment variables
export CODE_RAG_PATH="/path/to/didi"
export CODE_RAG_REPO_PATH="/path/to/degenduel-repo"
export CODE_RAG_DB_PATH="/path/to/database"

# Set custom model paths
export CODE_RAG_MODEL_PATH="codellama/CodeLlama-7b-instruct-hf"
export CODE_RAG_EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

## 💡 Examples

**Finding implementation details:**
```bash
./didi.sh search "user authentication flow"
```

**Getting detailed code snippets:**
```bash
./didi.sh details "websocket system architecture" -n 3
```

**Understanding complex patterns:**
```bash
./didi.sh ask "Explain how the contest system is implemented"
```

## 🏆 Features

- **Fast Semantic Search**: Find code based on concepts, not just literal text
- **Context-Aware Results**: Results include relevant surrounding code
- **DegenDuel-Specific Knowledge**: Tailored specifically for the DegenDuel codebase
- **Configurable Paths**: Works with any repository location
- **Optimized for Code**: Special handling for code files and structure
- **Expandable**: Didi can learn from codebase updates automatically

## 🛣️ Roadmap

- [ ] UI interface for easier interaction with Didi
- [ ] Integration with development environments
- [ ] Support for additional DegenDuel repositories
- [ ] Enhanced visualization of code relationships
- [ ] Function-level understanding and summarization

## 📚 How Didi Works

1. **Indexing**: Code files are processed, chunked, and embedded into vectors
2. **Storage**: Vectors are stored in a database for fast retrieval
3. **Search**: User queries are converted to vectors and matched against the database
4. **Retrieval**: Most relevant code is retrieved based on vector similarity
5. **Analysis**: Didi provides insights based on retrieved code

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- LlamaIndex for the RAG framework
- Sentence-Transformers for embeddings
- DegenDuel team for creating an awesome platform 