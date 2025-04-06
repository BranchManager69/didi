#!/bin/bash
# Didi with Ollama/Llama 4 - DegenDuel's AI Assistant
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
export DIDI_USE_OLLAMA="true"
export OLLAMA_MODEL="llama4"
export OLLAMA_URL="http://localhost:11434"

# Create required directories
mkdir -p "$CODE_RAG_REPOS_PATH"
mkdir -p "$CODE_RAG_DB_PATH"
mkdir -p "$(dirname $CODE_RAG_CONFIG_PATH)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
function print_header() {
    echo -e "\n${BLUE}==============================================${NC}"
    echo -e "${BLUE}     Didi with Llama 4 - DegenDuel's AI Assistant     ${NC}"
    echo -e "${BLUE}==============================================${NC}\n"
}

# Print usage information
function usage() {
    print_header
    echo -e "Usage: ./didi_ollama.sh [command] [arguments]"
    echo -e "\nSearch Commands:"
    echo -e "  ${GREEN}get${NC} [query]                 Find code matches (quick search)"
    echo -e "  ${GREEN}see${NC} [query]                 Alias for get"
    echo -e "  ${GREEN}g${NC} [query]                   Short alias for get"
    echo -e "  ${GREEN}details${NC} [query]             Get detailed code snippets"
    echo -e "  ${GREEN}d${NC} [query]                   Short alias for details"
    echo -e "  ${GREEN}ask${NC} [question]              Ask Didi questions about the code"
    echo -e "  ${GREEN}a${NC} [question]                Short alias for ask"
    echo -e "  ${GREEN}interactive${NC}                 Start interactive session with Didi"
    echo -e "  ${GREEN}i${NC}                           Short alias for interactive"
    echo -e "  ${GREEN}test${NC} [queries]              Run A/B tests on embedding models"
    
    echo -e "\nRepository Management:"
    echo -e "  ${GREEN}add-repo${NC} <n> <path/url>  Add a new repository to Didi"
    echo -e "  ${GREEN}list-repos${NC}                  List all repositories"
    echo -e "  ${GREEN}enable-repo${NC} <repo_key>      Enable a repository"
    echo -e "  ${GREEN}disable-repo${NC} <repo_key>     Disable a repository"
    echo -e "  ${GREEN}update${NC}                      Update all repos and rebuild if needed"
    
    echo -e "\nOllama Commands:"
    echo -e "  ${GREEN}model${NC} [model_name]          Switch to a different Ollama model"
    echo -e "  ${GREEN}m${NC} [model_name]              Short alias for model"
    
    echo -e "\nSystem Commands:"
    echo -e "  ${GREEN}setup${NC}                      Set up Didi's environment with Ollama"
    echo -e "  ${GREEN}status${NC}                     Check Didi's system status"
    echo -e "  ${GREEN}index${NC}                      Force rebuild of knowledge base"
    echo -e "  ${GREEN}parallel-index${NC}             Force rebuild with parallel processing"
    echo -e "  ${GREEN}help${NC}                       Show this help message"
    
    echo -e "\nExamples:"
    echo -e "  ./didi_ollama.sh get \"user auth\"             (Find code matches)"
    echo -e "  ./didi_ollama.sh ask \"How does websocket connection work?\" (Ask about code)"
    echo -e "  ./didi_ollama.sh interactive (Start interactive session with Llama 4)"
    echo -e "  ./didi_ollama.sh model llama3 (Switch to Llama 3 model)"
}

# Setup the environment
function setup_environment() {
    print_header
    echo -e "${YELLOW}Setting up Didi's environment with Ollama and Llama 4...${NC}\n"
    
    # Create directories if they don't exist
    mkdir -p "/home/ubuntu/degenduel-gpu/models"
    mkdir -p "/home/ubuntu/degenduel-gpu/venvs"
    mkdir -p "/home/ubuntu/degenduel-gpu/data/chroma_db"
    mkdir -p "/home/ubuntu/degenduel-gpu/config"
    mkdir -p "/home/ubuntu/degenduel-gpu/repos"
    
    # Run Ollama setup script
    echo -e "${YELLOW}Setting up Ollama with Llama 4...${NC}"
    bash "$SCRIPT_DIR/setup_ollama.sh"
    
    # Copy repos_config.json to persistent storage if it doesn't exist
    if [ ! -f "$CODE_RAG_CONFIG_PATH" ] && [ -f "$SCRIPT_DIR/repos_config.json" ]; then
        echo -e "${YELLOW}Copying repositories config to persistent storage...${NC}"
        cp "$SCRIPT_DIR/repos_config.json" "$CODE_RAG_CONFIG_PATH"
        echo -e "${GREEN}Repositories config copied to persistent storage.${NC}"
    fi
    
    # Copy repositories if they don't exist
    if [ -d "$SCRIPT_DIR/repos" ] && [ "$(ls -A $SCRIPT_DIR/repos 2>/dev/null)" ]; then
        if [ ! "$(ls -A $CODE_RAG_REPOS_PATH 2>/dev/null)" ]; then
            echo -e "${YELLOW}Copying repositories to persistent storage...${NC}"
            cp -r "$SCRIPT_DIR/repos"/* "$CODE_RAG_REPOS_PATH/"
            echo -e "${GREEN}Repositories copied to persistent storage.${NC}"
        fi
    fi
    
    # Copy ChromaDB if it doesn't exist
    if [ -d "$SCRIPT_DIR/chroma_db" ] && [ "$(ls -A $SCRIPT_DIR/chroma_db 2>/dev/null)" ]; then
        if [ ! "$(ls -A $CODE_RAG_DB_PATH 2>/dev/null)" ]; then
            echo -e "${YELLOW}Copying ChromaDB to persistent storage...${NC}"
            cp -r "$SCRIPT_DIR/chroma_db"/* "$CODE_RAG_DB_PATH/"
            echo -e "${GREEN}ChromaDB copied to persistent storage.${NC}"
        fi
    fi
    
    # Install Python dependencies
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r "$SCRIPT_DIR/requirements.txt"
    
    # Install Ollama Python client
    echo -e "${YELLOW}Installing Ollama Python client...${NC}"
    pip install ollama
    
    echo -e "\n${GREEN}Didi with Ollama/Llama 4 is now set up and ready to use!${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Run './didi_ollama.sh parallel-index' to build Didi's knowledge base"
    echo -e "  2. Run './didi_ollama.sh interactive' to start an interactive session with Didi using Llama 4"
}

# Index the repository
function index_repo() {
    print_header
    echo -e "${YELLOW}Building Didi's knowledge base...${NC}"
    echo -e "${YELLOW}This may take a few minutes depending on the size of the repositories.${NC}\n"
    
    python scripts/index_code.py
    
    echo -e "\n${GREEN}Knowledge base built successfully!${NC}"
    echo -e "You can now use 'get' or 'ask' commands."
}

# Index the repository with parallel processing
function parallel_index_repo() {
    print_header
    echo -e "${YELLOW}Building Didi's knowledge base with parallel processing...${NC}"
    echo -e "${YELLOW}This will be faster but may use more system resources.${NC}\n"
    
    if [ -f "$SCRIPT_DIR/scripts/parallel_index.py" ]; then
        python scripts/parallel_index.py
        echo -e "\n${GREEN}Knowledge base built successfully with parallel processing!${NC}"
    else
        echo -e "${RED}Parallel indexing script not found, using standard indexing...${NC}\n"
        python scripts/index_code.py
        echo -e "\n${GREEN}Knowledge base built successfully!${NC}"
    fi
    
    echo -e "You can now use 'get' or 'ask' commands."
}

# Search the repository
function search_repo() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Search query is required${NC}"
        echo -e "Usage: ./didi_ollama.sh get \"your search query\""
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
        echo -e "Usage: ./didi_ollama.sh details \"your search query\""
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
        echo -e "Usage: ./didi_ollama.sh ask \"your question about the code\""
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}Didi with Llama 4 is analyzing the code to answer: ${NC}$*\n"
    echo -e "${YELLOW}This may take a moment as Didi needs to think...${NC}\n"
    
    # Check if enhanced query script exists
    if [ -f "$SCRIPT_DIR/scripts/enhanced_query.py" ]; then
        python scripts/enhanced_query.py "$@"
    else
        python scripts/query_code.py "$@"
    fi
}

# Start interactive session
function start_interactive() {
    print_header
    echo -e "${YELLOW}Starting interactive session with Didi using Ollama/Llama 4...${NC}\n"
    
    # Check if enhanced query script exists
    if [ -f "$SCRIPT_DIR/scripts/enhanced_query.py" ]; then
        python scripts/enhanced_query.py -i
    else
        echo -e "${RED}Enhanced query script not found, switching to standard Q&A...${NC}\n"
        echo -e "${YELLOW}Please type your question:${NC}"
        read -p "> " question
        ask_question "$question"
    fi
}

# Update repositories and rebuild index only if needed
function update_repo() {
    print_header
    echo -e "${YELLOW}Updating repositories...${NC}\n"
    
    # Define a variable to track if any repo was updated
    REPO_UPDATED=false
    
    # Get list of repos from repos_config.json
    if [ -f "$CODE_RAG_CONFIG_PATH" ]; then
        REPOS=$(cat "$CODE_RAG_CONFIG_PATH" | grep -o '"path": "[^"]*"' | cut -d'"' -f4)
        
        for REPO_PATH in $REPOS; do
            if [ -d "$REPO_PATH" ] && [ -d "$REPO_PATH/.git" ]; then
                echo -e "${YELLOW}Updating repository at ${REPO_PATH}...${NC}"
                
                cd "$REPO_PATH"
                
                # Store the current commit hash
                BEFORE_HASH=$(git rev-parse HEAD)
                
                # Pull latest changes
                git pull
                
                # Get the new commit hash
                AFTER_HASH=$(git rev-parse HEAD)
                
                cd "$SCRIPT_DIR"
                
                # Check if this repo was updated
                if [ "$BEFORE_HASH" != "$AFTER_HASH" ]; then
                    echo -e "${GREEN}  ✓ Updated to new version${NC}"
                    REPO_UPDATED=true
                else
                    echo -e "${GREEN}  ✓ Already up to date${NC}"
                fi
            else
                # Get repo info from config to clone it if there's a git URL
                REPO_NAME=$(basename "$REPO_PATH")
                REPO_URL=$(cat "$CODE_RAG_CONFIG_PATH" | grep -A10 "\"path\": \"$REPO_PATH\"" | grep -o '"git_url": "[^"]*"' | head -n 1 | cut -d'"' -f4)
                
                if [ -n "$REPO_URL" ]; then
                    echo -e "${YELLOW}  ⟳ Repository not found. Cloning from $REPO_URL...${NC}"
                    mkdir -p "$(dirname "$REPO_PATH")"
                    git clone "$REPO_URL" "$REPO_PATH"
                    if [ $? -eq 0 ]; then
                        echo -e "${GREEN}  ✓ Repository cloned successfully${NC}"
                        REPO_UPDATED=true
                    else
                        echo -e "${RED}  ✗ Failed to clone repository${NC}"
                    fi
                else
                    echo -e "${RED}  ✗ Repository at ${REPO_PATH} is not a git repository or doesn't exist, and no git URL provided${NC}"
                fi
            fi
        done
    else
        echo -e "${RED}No repository configuration found in persistent storage.${NC}"
        echo -e "${YELLOW}Run './didi_ollama.sh setup' to initialize the environment.${NC}"
        exit 1
    fi
    
    echo -e "\n${GREEN}Repository update check complete.${NC}"
    
    # Only rebuild if there were actual changes
    if [ "$REPO_UPDATED" = true ]; then
        echo -e "${YELLOW}Changes detected! Didi is rebuilding its knowledge base...${NC}\n"
        
        # Check if parallel indexing script exists
        if [ -f "$SCRIPT_DIR/scripts/parallel_index.py" ]; then
            python scripts/parallel_index.py
        else
            python scripts/index_code.py
        fi
        
        echo -e "\n${GREEN}Didi's knowledge base updated successfully!${NC}"
    else
        echo -e "${GREEN}No changes detected. No need to rebuild the knowledge base.${NC}"
    fi
}

# Check status of the system
function check_status() {
    print_header
    echo -e "${YELLOW}Checking Didi's system status with Ollama/Llama 4...${NC}\n"
    
    # Check Ollama status
    echo -e "${BLUE}Ollama:${NC}"
    if command -v ollama &> /dev/null; then
        echo -e "  Ollama: ${GREEN}Installed${NC}"
        if pgrep -x "ollama" > /dev/null; then
            echo -e "  Ollama Service: ${GREEN}Running${NC}"
            # Check if Llama 4 model is installed
            if ollama list 2>/dev/null | grep -q "llama4"; then
                echo -e "  Llama 4 Model: ${GREEN}Installed${NC}"
            else
                echo -e "  Llama 4 Model: ${RED}Not installed${NC}"
                echo -e "  Run './didi_ollama.sh setup' to install Llama 4 model"
            fi
        else
            echo -e "  Ollama Service: ${RED}Not running${NC}"
            echo -e "  Run 'ollama serve' to start the Ollama service"
        fi
    else
        echo -e "  Ollama: ${RED}Not installed${NC}"
        echo -e "  Run './didi_ollama.sh setup' to install Ollama"
    fi
    
    # Check persistent storage
    echo -e "\n${BLUE}Persistent Storage:${NC}"
    if [ -d "/home/ubuntu/degenduel-gpu" ]; then
        echo -e "  Status: ${GREEN}Found${NC}"
        echo -e "  Model cache: ${GREEN}$(du -sh /home/ubuntu/degenduel-gpu/models 2>/dev/null | cut -f1 || echo "Empty")${NC}"
        echo -e "  Data storage: ${GREEN}$(du -sh /home/ubuntu/degenduel-gpu/data 2>/dev/null | cut -f1 || echo "Empty")${NC}"
    else
        echo -e "  Status: ${RED}Not found${NC}"
        echo -e "  Warning: Persistent storage not set up correctly!"
    fi
    
    # Check repositories
    echo -e "\n${BLUE}Repositories:${NC}"
    if [ -f "$CODE_RAG_CONFIG_PATH" ]; then
        # Extract info from repos_config.json using grep and sed
        REPO_KEYS=$(cat "$CODE_RAG_CONFIG_PATH" | grep -o '"[^"]*": {' | sed 's/": {//' | sed 's/"//g')
        
        for REPO_KEY in $REPO_KEYS; do
            REPO_NAME=$(cat "$CODE_RAG_CONFIG_PATH" | grep -A10 "\"$REPO_KEY\":" | grep "\"name\":" | head -1 | cut -d'"' -f4)
            REPO_PATH=$(cat "$CODE_RAG_CONFIG_PATH" | grep -A10 "\"$REPO_KEY\":" | grep "\"path\":" | head -1 | cut -d'"' -f4)
            REPO_ENABLED=$(cat "$CODE_RAG_CONFIG_PATH" | grep -A10 "\"$REPO_KEY\":" | grep "\"enabled\":" | head -1 | grep -o "true\|false")
            
            if [ "$REPO_ENABLED" = "true" ]; then
                STATUS="${GREEN}Enabled${NC}"
            else
                STATUS="${RED}Disabled${NC}"
            fi
            
            echo -e "  ${GREEN}$REPO_NAME${NC} ($REPO_KEY)"
            if [ -d "$REPO_PATH" ]; then
                echo -e "    Status: ${GREEN}Found${NC} | $STATUS"
                TS_FILES=$(find "$REPO_PATH" -type f -name "*.ts*" 2>/dev/null | wc -l)
                PY_FILES=$(find "$REPO_PATH" -type f -name "*.py" 2>/dev/null | wc -l)
                JS_FILES=$(find "$REPO_PATH" -type f -name "*.js*" 2>/dev/null | wc -l)
                
                if [ "$TS_FILES" -gt 0 ]; then
                    echo -e "    TypeScript files: ${GREEN}$TS_FILES${NC}"
                fi
                if [ "$PY_FILES" -gt 0 ]; then
                    echo -e "    Python files: ${GREEN}$PY_FILES${NC}"
                fi
                if [ "$JS_FILES" -gt 0 ]; then
                    echo -e "    JavaScript files: ${GREEN}$JS_FILES${NC}"
                fi
            else
                echo -e "    Status: ${RED}Not found${NC}"
            fi
            echo
        done
    else
        echo -e "  ${RED}No repositories configured${NC}"
    fi
    
    # Check index
    echo -e "${BLUE}Knowledge Base:${NC}"
    if [ -d "$CODE_RAG_DB_PATH" ]; then
        echo -e "  Status: ${GREEN}Found${NC}"
        DB_SIZE=$(du -sh "$CODE_RAG_DB_PATH" 2>/dev/null | cut -f1)
        echo -e "  Size: ${GREEN}$DB_SIZE${NC}"
        
        # Check if repos metadata exists
        if [ -f "$CODE_RAG_DB_PATH/repos_metadata.json" ]; then
            INDEXED_REPOS=$(cat "$CODE_RAG_DB_PATH/repos_metadata.json" | grep -o '"name": "[^"]*"' | cut -d'"' -f4 | tr '\n' ', ' | sed 's/, $//')
            echo -e "  Indexed repositories: ${GREEN}$INDEXED_REPOS${NC}"
        fi
    else
        echo -e "  Status: ${RED}Not built${NC}"
    fi
    
    # Check Python environment
    echo -e "\n${BLUE}Environment:${NC}"
    echo -e "  Python version: ${GREEN}$(python --version)${NC}"
    echo -e "  CUDA available: ${GREEN}$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "No")${NC}"
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        echo -e "  GPU: ${GREEN}$GPU_NAME${NC}"
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Unknown")
        echo -e "  CUDA version: ${GREEN}$CUDA_VERSION${NC}"
    fi
    
    echo -e "\n${BLUE}Required Python packages:${NC}"
    pip list | grep -E "llama-index|chroma|sentence-transformers|transformers|torch|ollama"
}

# Add function for using different Ollama models
function switch_model() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Model name is required${NC}"
        echo -e "Usage: ./didi_ollama.sh model llama4"
        exit 1
    fi
    
    MODEL_NAME="$1"
    
    print_header
    echo -e "${YELLOW}Switching to Ollama model: ${GREEN}$MODEL_NAME${NC}\n"
    
    # Check if model is available
    if ! ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
        echo -e "${YELLOW}Model $MODEL_NAME not found. Attempting to pull it...${NC}"
        ollama pull "$MODEL_NAME"
    fi
    
    # Update the environment variable in the script
    sed -i "s/export OLLAMA_MODEL=\"[^\"]*\"/export OLLAMA_MODEL=\"$MODEL_NAME\"/" "$SCRIPT_DIR/didi_ollama.sh"
    
    echo -e "${GREEN}Model switched to $MODEL_NAME successfully!${NC}"
    echo -e "${YELLOW}To use this model, run:${NC}"
    echo -e "  ./didi_ollama.sh interactive"
}

# Main command processing
if [ $# -eq 0 ]; then
    usage
    exit 0
fi

case "$1" in
    setup)
        setup_environment
        ;;
    index)
        index_repo
        ;;
    parallel-index)
        parallel_index_repo
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
    interactive|i)
        start_interactive
        ;;
    model|m)
        shift
        switch_model "$@"
        ;;
    test)
        shift
        run_tests "$@"
        ;;
    update)
        update_repo
        ;;
    status)
        check_status
        ;;
    add-repo)
        shift
        add_repo "$@"
        ;;
    list-repos)
        list_repos
        ;;
    enable-repo)
        shift
        toggle_repo "$@" "enable"
        ;;
    disable-repo)
        shift
        toggle_repo "$@" "disable"
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