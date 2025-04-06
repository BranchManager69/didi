#\!/bin/bash
# Didi - DegenDuel's AI Assistant
# ----------------------------------------------------

set -e

# Set up environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Export environment variables for Python scripts
export CODE_RAG_PATH="$SCRIPT_DIR"
export CODE_RAG_REPOS_PATH="/home/ubuntu/degenduel-gpu/repos"
export CODE_RAG_DB_PATH="/home/ubuntu/degenduel-gpu/data/chroma_db"
export CODE_RAG_CONFIG_PATH="$SCRIPT_DIR/repos_config.json"
export HF_HOME="/home/ubuntu/degenduel-gpu/models"
export TORCH_HOME="/home/ubuntu/degenduel-gpu/models"
# CUDA
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.0"

# Create required directories
mkdir -p "$CODE_RAG_REPOS_PATH"
mkdir -p "$CODE_RAG_DB_PATH"
mkdir -p "$(dirname $CODE_RAG_CONFIG_PATH)"

# Create virtual environment if it doesn't exist
if [ \! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

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
    echo -e "\n${BLUE}Model Profile Management:${NC}"
    echo -e "  ${GREEN}profile${NC}             List available model profiles"
    echo -e "  ${GREEN}profile switch${NC} [name]   Switch to a different model profile"
    echo -e "  ${GREEN}profile create${NC} [args]   Create a new model profile"
    echo -e "\nExamples:"
    echo -e "  ./didi.sh get \"user auth\"      (Find code matches)"
    echo -e "  ./didi.sh ask \"How does it work?\" (Ask for explanation)"
    echo -e "  ./didi.sh profile switch powerful  (Switch to powerful model profile)"
}

# Index the repository
function index_repo() {
    print_header
    echo -e "${YELLOW}Building Didi's knowledge base...${NC}"
    echo -e "${YELLOW}This may take a few minutes depending on the size of the repository.${NC}\n"
    
    python scripts/index_code.py
    
    echo -e "\n${GREEN}Knowledge base built successfully\!${NC}"
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
    
    cd "$CODE_RAG_REPOS_PATH"
    for dir in */; do
        if [ -d "$dir/.git" ]; then
            echo -e "${BLUE}Updating ${dir%/}...${NC}"
            (cd "$dir" && git pull)
        fi
    done
    cd "$SCRIPT_DIR"
    
    echo -e "\n${GREEN}Repositories updated\!${NC}"
    echo -e "${YELLOW}Didi is rebuilding its knowledge base...${NC}\n"
    
    python scripts/index_code.py
    
    echo -e "\n${GREEN}Didi's knowledge base updated successfully\!${NC}"
}

# Check status of the system
function check_status() {
    print_header
    echo -e "${YELLOW}Checking Didi's system status...${NC}\n"
    
    # Check persistent storage
    echo -e "${BLUE}Persistent Storage:${NC}"
    if [ -d "/home/ubuntu/degenduel-gpu" ]; then
        echo -e "  Status: ${GREEN}Found${NC}"
        MODEL_CACHE_SIZE=$(du -sh "/home/ubuntu/degenduel-gpu/models" 2>/dev/null | cut -f1 || echo "Not available")
        echo -e "  Model cache: ${GREEN}$MODEL_CACHE_SIZE${NC}"
        DATA_SIZE=$(du -sh "/home/ubuntu/degenduel-gpu/data" 2>/dev/null | cut -f1 || echo "Not available")
        echo -e "  Data storage: ${GREEN}$DATA_SIZE${NC}"
    else
        echo -e "  Status: ${RED}Not found${NC}"
    fi
    echo -e ""
    
    # Check model configuration
    echo -e "${BLUE}AI Models:${NC}"
    EMBED_MODEL=$(grep "DEFAULT_EMBED_MODEL.*sentence-transformers" "$SCRIPT_DIR/config.py" | grep -o '"[^"]*"' | tail -n1 | tr -d '"')
    if [ -z "$EMBED_MODEL" ]; then
        EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
    fi
    echo -e "  Embedding model: ${GREEN}$EMBED_MODEL${NC}"
    
    LLM_MODEL=$(grep "DEFAULT_MODEL_PATH.*CodeLlama" "$SCRIPT_DIR/config.py" | grep -o '"[^"]*"' | tail -n1 | tr -d '"')
    if [ -z "$LLM_MODEL" ]; then
        LLM_MODEL="codellama/CodeLlama-7b-instruct-hf"
    fi
    echo -e "  LLM model: ${GREEN}$LLM_MODEL${NC}"
    
    if command -v ollama &> /dev/null; then
        echo -e "  Ollama: ${GREEN}Installed${NC}"
        OLLAMA_MODELS=$(ollama list 2>/dev/null | grep -v "NAME" | awk '{print $1}' | tr '\n' ', ' | sed 's/, $//' || echo "None")
        echo -e "  Available Ollama models: ${GREEN}$OLLAMA_MODELS${NC}"
    else
        echo -e "  Ollama: ${RED}Not installed${NC}"
    fi
    echo -e ""
    
    # Check repositories
    echo -e "${BLUE}Repositories:${NC}"
    if [ -f "$CODE_RAG_CONFIG_PATH" ]; then
        cat "$CODE_RAG_CONFIG_PATH" | grep -o '"name": "[^"]*"' | cut -d'"' -f4 | while read -r repo_name; do
            repo_key=$(cat "$CODE_RAG_CONFIG_PATH" | grep -B 2 "\"name\": \"$repo_name\"" | grep -o '"[^"]*": {' | head -n 1 | cut -d'"' -f2)
            repo_path=$(cat "$CODE_RAG_CONFIG_PATH" | grep -A 5 "\"name\": \"$repo_name\"" | grep "path" | cut -d'"' -f4)
            enabled=$(cat "$CODE_RAG_CONFIG_PATH" | grep -A 5 "\"name\": \"$repo_name\"" | grep "enabled" | grep -o "true\|false")
            
            echo -e "  ${GREEN}$repo_name${NC} ($repo_key)"
            if [ -d "$repo_path" ]; then
                echo -e "    Status: ${GREEN}Found${NC} | ${GREEN}${enabled^}${NC}"
                
                # Check file types
                TYPESCRIPT_FILES=$(find "$repo_path" -name "*.ts*" 2>/dev/null | wc -l)
                PYTHON_FILES=$(find "$repo_path" -name "*.py" 2>/dev/null | wc -l)
                JAVASCRIPT_FILES=$(find "$repo_path" -name "*.js" 2>/dev/null | wc -l)
                
                if [ "$TYPESCRIPT_FILES" -gt 0 ]; then
                    echo -e "    TypeScript files: ${GREEN}$TYPESCRIPT_FILES${NC}"
                fi
                if [ "$PYTHON_FILES" -gt 0 ]; then
                    echo -e "    Python files: ${GREEN}$PYTHON_FILES${NC}"
                fi
                if [ "$JAVASCRIPT_FILES" -gt 0 ]; then
                    echo -e "    JavaScript files: ${GREEN}$JAVASCRIPT_FILES${NC}"
                fi
            else
                echo -e "    Status: ${RED}Not found${NC} | ${enabled^}"
            fi
            echo -e ""
        done
    else
        echo -e "  ${RED}Repository configuration not found${NC}"
    fi
    
    # Check index
    echo -e "${BLUE}Knowledge Base:${NC}"
    if [ -d "$CODE_RAG_DB_PATH" ]; then
        echo -e "  Status: ${GREEN}Found${NC}"
        DB_SIZE=$(du -sh "$CODE_RAG_DB_PATH" 2>/dev/null | cut -f1)
        echo -e "  Size: ${GREEN}$DB_SIZE${NC}"
        
        if [ -f "$CODE_RAG_DB_PATH/repos_metadata.json" ]; then
            INDEXED_REPOS=$(cat "$CODE_RAG_DB_PATH/repos_metadata.json" | grep -o '"name": "[^"]*"' | cut -d'"' -f4 | tr '\n' ', ' | sed 's/, $//')
            echo -e "  Indexed repositories: ${GREEN}$INDEXED_REPOS${NC}"
        fi
    else
        echo -e "  Status: ${RED}Not built${NC}"
    fi
    echo -e ""
    
    # Check Docker
    echo -e "${BLUE}Docker:${NC}"
    if command -v docker &> /dev/null; then
        echo -e "  Docker: ${GREEN}Installed${NC}"
    else
        echo -e "  Docker: ${RED}Not installed${NC}"
    fi
    if command -v docker-compose &> /dev/null; then
        echo -e "  Docker Compose: ${GREEN}Installed${NC}"
    else
        echo -e "  Docker Compose: ${RED}Not installed${NC}"
    fi
    echo -e ""
    
    # Check environment
    echo -e "${BLUE}Environment:${NC}"
    echo -e "  Python version: ${GREEN}$(python --version)${NC}"
    echo -e "  CUDA available: ${GREEN}$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "No")${NC}"
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        echo -e "  GPU: ${GREEN}$GPU_NAME${NC}"
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Unknown")
        echo -e "  CUDA version: ${GREEN}$CUDA_VERSION${NC}"
    fi
    echo -e ""
    
    echo -e "${BLUE}Required Python packages:${NC}"
    pip list | grep -E "llama-index|chroma|sentence-transformers|transformers|torch"
}

# Profile management functions
function manage_profiles() {
    if [ $# -eq 0 ]; then
        python "$SCRIPT_DIR/scripts/model_profile.py" list
        return
    fi
    
    case "$1" in
        list)
            python "$SCRIPT_DIR/scripts/model_profile.py" list
            ;;
        switch)
            if [ $# -lt 2 ]; then
                echo -e "${RED}Error: Profile name is required${NC}"
                echo -e "Usage: ./didi.sh profile switch [profile_name]"
                return 1
            fi
            python "$SCRIPT_DIR/scripts/model_profile.py" switch "$2"
            # Source the switch script if it exists
            switch_script="$SCRIPT_DIR/model_profiles/switch_profile.sh"
            if [ -f "$switch_script" ]; then
                source "$switch_script"
                echo -e "${GREEN}Profile activated.${NC} Run ${YELLOW}didi status${NC} to confirm changes."
            fi
            ;;
        create)
            shift
            python "$SCRIPT_DIR/scripts/model_profile.py" create "$@"
            ;;
        reset)
            python "$SCRIPT_DIR/scripts/model_profile.py" reset
            ;;
        *)
            echo -e "${RED}Unknown profile command: $1${NC}"
            echo -e "Available commands: list, switch, create, reset"
            return 1
            ;;
    esac
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
    profile|p)
        shift
        manage_profiles "$@"
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
