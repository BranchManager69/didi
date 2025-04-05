#!/bin/bash
# Didi - DegenDuel's AI Assistant
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
    echo -e "${BLUE}           Didi - DegenDuel's AI Assistant      ${NC}"
    echo -e "${BLUE}==============================================${NC}\n"
}

# Print usage information
function usage() {
    print_header
    echo -e "Usage: ./didi.sh [command] [arguments]"
    echo -e "\nCommands:"
    echo -e "  ${GREEN}get${NC} [query]         Find code matches (quick search)"
    echo -e "  ${GREEN}see${NC} [query]         Alias for get"
    echo -e "  ${GREEN}g${NC} [query]           Short alias for get"
    echo -e "  ${GREEN}details${NC} [query]     Get detailed code snippets"
    echo -e "  ${GREEN}d${NC} [query]           Short alias for details"
    echo -e "  ${GREEN}ask${NC} [question]      Ask Didi questions about the code"
    echo -e "  ${GREEN}a${NC} [question]        Short alias for ask"
    echo -e "  ${GREEN}update${NC}              Update repo and rebuild knowledge"
    echo -e "  ${GREEN}status${NC}              Check Didi's system status"
    echo -e "  ${GREEN}help${NC}                Show this help message"
    echo -e "\nExamples:"
    echo -e "  ./didi.sh get \"user auth\"      (Find code matches)"
    echo -e "  ./didi.sh see \"websocket\"      (Same as get)"
    echo -e "  ./didi.sh ask \"How does it work?\" (Ask for explanation)"
}

# Index the repository
function index_repo() {
    print_header
    echo -e "${YELLOW}Building Didi's knowledge base...${NC}"
    echo -e "${YELLOW}This may take a few minutes depending on the size of the repository.${NC}\n"
    
    python scripts/index_code.py
    
    echo -e "\n${GREEN}Knowledge base built successfully!${NC}"
    echo -e "You can now use 'get' or 'ask' commands."
}

# Search the repository
function search_repo() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Search query is required${NC}"
        echo -e "Usage: ./didi.sh get \"your search query\""
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}Didi is finding code for: ${NC}$*\n"
    
    python scripts/search_code.py "$@"
}

# Get detailed code snippets
function details_repo() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Search query is required${NC}"
        echo -e "Usage: ./didi.sh details \"your search query\""
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}Didi is finding detailed code for: ${NC}$*\n"
    
    python scripts/simple_search.py "$@"
}

# Ask questions about the repository
function ask_question() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Question is required${NC}"
        echo -e "Usage: ./didi.sh ask \"your question about the code\""
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}Didi is analyzing the code to answer: ${NC}$*\n"
    echo -e "${YELLOW}This may take a moment as Didi needs to think...${NC}\n"
    
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
    echo -e "${YELLOW}Didi is rebuilding its knowledge base...${NC}\n"
    
    python scripts/index_code.py
    
    echo -e "\n${GREEN}Didi's knowledge base updated successfully!${NC}"
}

# Check status of the system
function check_status() {
    print_header
    echo -e "${YELLOW}Checking Didi's system status...${NC}\n"
    
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
        echo -e "Knowledge base: ${GREEN}Found${NC}"
        DB_SIZE=$(du -sh "$CODE_RAG_DB_PATH" | cut -f1)
        echo -e "Knowledge size: ${GREEN}$DB_SIZE${NC}"
    else
        echo -e "Knowledge base: ${RED}Not built${NC}"
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
    get|see|g|search|s)
        shift
        search_repo "$@"
        ;;
    details|d)
        shift
        details_repo "$@"
        ;;
    ask|a)
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