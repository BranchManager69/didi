# CodeRAG: Code Repository Analysis with LLMs

A powerful Retrieval-Augmented Generation (RAG) system for semantic code search and understanding.

![CodeRAG Demo](https://i.ibb.co/f2smYQH/coderag-diagram.png)

## 📋 Overview

CodeRAG transforms your code repository into a searchable knowledge base, enabling:

- 🔍 **Semantic Code Search**: Find relevant code based on meaning, not just keywords
- 📝 **Detailed Code Exploration**: Get full context and implementation details
- 🤖 **LLM-Powered Insights**: (optional) Ask complex questions about your codebase
- 🚀 **Simple Command Interface**: Easy-to-use shell commands for all functionality

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Git
- A code repository to analyze

### Setup

```bash
# Clone this repository
git clone https://github.com/yourusername/coderag.git
cd coderag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Clone a repository to analyze
git clone https://github.com/your/target-repo.git repo

# Build the index
./coderag.sh index
```

## 🚀 Usage

CodeRAG provides a simple command-line interface for all functionality:

```bash
# Show available commands
./coderag.sh help

# Check system status
./coderag.sh status

# Search code
./coderag.sh search "websocket authentication"

# Detailed code view
./coderag.sh details "contest system implementation"

# Ask questions about the code (requires LLM)
./coderag.sh ask "How does the user authentication work?"

# Update repository and rebuild index
./coderag.sh update
```

## 🔧 Configuration

CodeRAG can be configured via environment variables or by editing `config.py`:

```bash
# Set custom paths via environment variables
export CODE_RAG_PATH="/path/to/coderag"
export CODE_RAG_REPO_PATH="/path/to/repo"
export CODE_RAG_DB_PATH="/path/to/database"

# Set custom model paths
export CODE_RAG_MODEL_PATH="codellama/CodeLlama-7b-instruct-hf"
export CODE_RAG_EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

## 💡 Examples

**Finding implementation details:**
```bash
./coderag.sh search "user authentication flow"
```

**Getting detailed code snippets:**
```bash
./coderag.sh details "websocket system architecture" -n 3
```

**Understanding complex patterns:**
```bash
./coderag.sh ask "Explain how the contest system is implemented"
```

## 🏆 Features

- **Fast Semantic Search**: Find code based on concepts, not just literal text
- **Context-Aware Results**: Results include relevant surrounding code
- **Lightweight Embedding**: Works without needing a full LLM for basic search
- **Configurable Paths**: Works with any repository location
- **Optimized for Code**: Special handling for code files and structure
- **Expandable**: Add your own search patterns or customize the LLM

## 🛣️ Roadmap

- [ ] UI interface for easier browsing
- [ ] Integration with development environments
- [ ] Support for more programming languages
- [ ] Enhanced visualization of code relationships
- [ ] Function-level understanding and summarization

## 📚 How It Works

1. **Indexing**: Code files are processed, chunked, and embedded into vectors
2. **Storage**: Vectors are stored in a ChromaDB database for fast retrieval
3. **Search**: User queries are converted to vectors and matched against the database
4. **Retrieval**: Most relevant code is retrieved based on vector similarity
5. **Analysis**: (optional) LLM provides insights based on retrieved code

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- LlamaIndex for the RAG framework
- Sentence-Transformers for embeddings
- DegenDuel for the example repository 