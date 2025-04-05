#!/bin/bash
# CodeRAG - Easy interface for code repository analysis
# ----------------------------------------------------

set -e

# Set up environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Export environment variables for Python scripts
export CODE_RAG_PATH="$SCRIPT_DIR"
export CODE_RAG_REPO_PATH="$SCRIPT_DIR/repo"
export CODE_RAG_DB_PATH="$SCRIPT_DIR/chroma_db"

# Activate virtual environment
source venv/bin/activate

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
function print_header() {
    echo -e "\n${BLUE}==============================================${NC}"
    echo -e "${BLUE}           DegenDuel Code RAG System           ${NC}"
    echo -e "${BLUE}==============================================${NC}\n"
}

# Print usage information
function usage() {
    print_header
    echo -e "Usage: ./coderag.sh [command] [arguments]"
    echo -e "\nCommands:"
    echo -e "  ${GREEN}index${NC}               Build the vector index of the repository"
    echo -e "  ${GREEN}search${NC} [query]      Semantic search for code (fast)"
    echo -e "  ${GREEN}details${NC} [query]     Get detailed code snippets for your query"
    echo -e "  ${GREEN}ask${NC} [question]      Ask questions about the codebase (uses LLM)"
    echo -e "  ${GREEN}update${NC}              Update the repository and rebuild index"
    echo -e "  ${GREEN}status${NC}              Check the status of the RAG system"
    echo -e "  ${GREEN}help${NC}                Show this help message"
    echo -e "\nExamples:"
    echo -e "  ./coderag.sh search \"user authentication flow\""
    echo -e "  ./coderag.sh details \"websocket architecture\""
    echo -e "  ./coderag.sh ask \"How does the contest system work?\""
    echo -e "  ./coderag.sh index"
}

# Index the repository
function index_repo() {
    print_header
    echo -e "${YELLOW}Building code index...${NC}"
    echo -e "${YELLOW}This may take a few minutes depending on the size of the repository.${NC}\n"
    
    python scripts/index_code.py
    
    echo -e "\n${GREEN}Index built successfully!${NC}"
    echo -e "You can now use 'search' or 'ask' commands."
}

# Search the repository
function search_repo() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Search query is required${NC}"
        echo -e "Usage: ./coderag.sh search \"your search query\""
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}Searching code for: ${NC}$*\n"
    
    python scripts/search_code.py "$@"
}

# Get detailed code snippets
function details_repo() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Search query is required${NC}"
        echo -e "Usage: ./coderag.sh details \"your search query\""
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}Getting detailed code for: ${NC}$*\n"
    
    python scripts/simple_search.py "$@"
}

# Ask questions about the repository
function ask_question() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Question is required${NC}"
        echo -e "Usage: ./coderag.sh ask \"your question about the code\""
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}Analyzing codebase to answer: ${NC}$*\n"
    echo -e "${YELLOW}This may take a moment as it needs to load the model...${NC}\n"
    
    python scripts/query_code.py "$@"
}

# Update the repository and rebuild index
function update_repo() {
    print_header
    echo -e "${YELLOW}Updating repository...${NC}\n"
    
    cd repo
    git pull
    cd ..
    
    echo -e "\n${GREEN}Repository updated!${NC}"
    echo -e "${YELLOW}Rebuilding index...${NC}\n"
    
    python scripts/index_code.py
    
    echo -e "\n${GREEN}Index updated successfully!${NC}"
}

# Check status of the RAG system
function check_status() {
    print_header
    echo -e "${YELLOW}Checking RAG system status...${NC}\n"
    
    # Check repo
    if [ -d "$CODE_RAG_REPO_PATH" ]; then
        echo -e "Repository: ${GREEN}Found${NC}"
        REPO_FILES=$(find "$CODE_RAG_REPO_PATH" -type f -name "*.ts*" | wc -l)
        echo -e "TypeScript files: ${GREEN}$REPO_FILES${NC}"
    else
        echo -e "Repository: ${RED}Not found${NC}"
    fi
    
    # Check index
    if [ -d "$CODE_RAG_DB_PATH" ]; then
        echo -e "Vector index: ${GREEN}Found${NC}"
        DB_SIZE=$(du -sh "$CODE_RAG_DB_PATH" | cut -f1)
        echo -e "Index size: ${GREEN}$DB_SIZE${NC}"
    else
        echo -e "Vector index: ${RED}Not built${NC}"
    fi
    
    # Check Python environment
    echo -e "Python version: ${GREEN}$(python --version)${NC}"
    
    echo -e "\n${BLUE}Required Python packages:${NC}"
    pip list | grep -E "llama-index|langchain|chroma|sentence-transformers|transformers|torch"
}

# Main command processing
if [ $# -eq 0 ]; then
    usage
    exit 0
fi

case "$1" in
    index)
        index_repo
        ;;
    search)
        shift
        search_repo "$@"
        ;;
    details)
        shift
        details_repo "$@"
        ;;
    ask)
        shift
        ask_question "$@"
        ;;
    update)
        update_repo
        ;;
    status)
        check_status
        ;;
    help)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        usage
        exit 1
        ;;
esac

exit 0 