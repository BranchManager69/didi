#!/bin/bash
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
export CODE_RAG_CONFIG_PATH="/home/ubuntu/degenduel-gpu/config/repos_config.json"
export HF_HOME="/home/ubuntu/degenduel-gpu/models"
export TORCH_HOME="/home/ubuntu/degenduel-gpu/models"

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
    echo -e "${BLUE}           Didi - DegenDuel's AI Assistant      ${NC}"
    echo -e "${BLUE}==============================================${NC}\n"
}

# Print usage information
function usage() {
    print_header
    echo -e "Usage: ./didi.sh [command] [arguments]"
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
    
    echo -e "\nSystem Commands:"
    echo -e "  ${GREEN}setup${NC}                      Set up Didi's environment"
    echo -e "  ${GREEN}status${NC}                     Check Didi's system status"
    echo -e "  ${GREEN}index${NC}                      Force rebuild of knowledge base"
    echo -e "  ${GREEN}parallel-index${NC}             Force rebuild with parallel processing"
    echo -e "  ${GREEN}docker${NC}                     Run Didi in a Docker container"
    echo -e "  ${GREEN}help${NC}                       Show this help message"
    
    echo -e "\nExamples:"
    echo -e "  ./didi.sh get \"user auth\"             (Find code matches)"
    echo -e "  ./didi.sh ask \"How does websocket connection work?\" (Ask about code)"
    echo -e "  ./didi.sh add-repo \"MyProject\" /path/to/repo  (Add local repo)"
    echo -e "  ./didi.sh add-repo \"MyProject\" https://github.com/user/repo.git (Add remote repo)"
    echo -e "  ./didi.sh interactive (Start interactive session)"
}

# Setup the environment
function setup_environment() {
    print_header
    echo -e "${YELLOW}Setting up Didi's environment...${NC}\n"
    
    # Create directories if they don't exist
    mkdir -p "/home/ubuntu/degenduel-gpu/models"
    mkdir -p "/home/ubuntu/degenduel-gpu/venvs"
    mkdir -p "/home/ubuntu/degenduel-gpu/data/chroma_db"
    mkdir -p "/home/ubuntu/degenduel-gpu/config"
    mkdir -p "/home/ubuntu/degenduel-gpu/repos"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}Docker not found, installing...${NC}"
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        echo -e "${GREEN}Docker installed. You may need to log out and back in for group changes to take effect.${NC}"
    else
        echo -e "${GREEN}Docker is already installed.${NC}"
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${YELLOW}Docker Compose not found, installing...${NC}"
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.6/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        echo -e "${GREEN}Docker Compose installed.${NC}"
    else
        echo -e "${GREEN}Docker Compose is already installed.${NC}"
    fi
    
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
    
    echo -e "\n${GREEN}Didi's environment is now set up and ready to use!${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Run './didi.sh parallel-index' to build Didi's knowledge base"
    echo -e "  2. Run './didi.sh interactive' to start an interactive session with Didi"
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
    echo -e "${YELLOW}Starting interactive session with Didi...${NC}\n"
    
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

# Run Didi in Docker
function run_docker() {
    print_header
    echo -e "${YELLOW}Running Didi in Docker...${NC}\n"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found. Please run './didi.sh setup' first.${NC}"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose not found. Please run './didi.sh setup' first.${NC}"
        exit 1
    fi
    
    # Run Docker Compose
    cd "$SCRIPT_DIR"
    docker-compose up --build
}

# Run A/B tests on embedding models
function run_tests() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Test queries are required${NC}"
        echo -e "Usage: ./didi.sh test \"query1\" \"query2\" ..."
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}Didi is running A/B tests on embedding models...${NC}"
    echo -e "${YELLOW}Testing queries: ${NC}$*\n"
    
    # Pass all queries to the test script
    if [ -f "$SCRIPT_DIR/scripts/ab_test_embeddings.py" ]; then
        python scripts/ab_test_embeddings.py --queries "$@"
    else
        python scripts/ab_test_demo.py "$@"
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
                echo -e "${RED}  ✗ Repository at ${REPO_PATH} is not a git repository or doesn't exist${NC}"
            fi
        done
    else
        echo -e "${RED}No repository configuration found in persistent storage.${NC}"
        echo -e "${YELLOW}Run './didi.sh setup' to initialize the environment.${NC}"
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

# Add a new repository to the system
function add_repo() {
    if [ $# -lt 2 ]; then
        echo -e "${RED}Error: Repository name and path/URL are required${NC}"
        echo -e "Usage: ./didi.sh add-repo <n> <path_or_url> [description]"
        exit 1
    fi
    
    print_header
    REPO_KEY=$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    REPO_NAME="$1"
    REPO_PATH_OR_URL="$2"
    REPO_DESC="${3:-A codebase added to Didi}"
    
    # Determine if it's a local path or git URL
    if [[ "$REPO_PATH_OR_URL" == "http"* || "$REPO_PATH_OR_URL" == "git@"* ]]; then
        # It's a git URL, clone it
        REPO_PATH="$CODE_RAG_REPOS_PATH/$REPO_KEY"
        REPO_URL="$REPO_PATH_OR_URL"
        
        echo -e "${YELLOW}Cloning repository from $REPO_URL to $REPO_PATH...${NC}\n"
        
        if [ -d "$REPO_PATH" ]; then
            echo -e "${RED}Directory already exists at $REPO_PATH${NC}"
            echo -e "${YELLOW}Updating existing repository instead...${NC}\n"
            cd "$REPO_PATH"
            git pull
            cd "$SCRIPT_DIR"
        else
            mkdir -p "$CODE_RAG_REPOS_PATH"
            git clone "$REPO_URL" "$REPO_PATH"
        fi
    else
        # It's a local path
        REPO_PATH="$REPO_PATH_OR_URL"
        REPO_URL=""
        
        echo -e "${YELLOW}Using existing repository at $REPO_PATH...${NC}\n"
        
        if [ ! -d "$REPO_PATH" ]; then
            echo -e "${RED}Error: Directory $REPO_PATH does not exist${NC}"
            exit 1
        fi
    fi
    
    # Update repos_config.json
    TMP_CONFIG=$(mktemp)
    
    if [ -f "$CODE_RAG_CONFIG_PATH" ]; then
        # Extract current config
        cat "$CODE_RAG_CONFIG_PATH" > "$TMP_CONFIG"
        # Remove closing brace
        sed -i '$ d' "$TMP_CONFIG"
        # If not empty, add a comma
        FILESIZE=$(wc -c < "$TMP_CONFIG")
        if [ "$FILESIZE" -gt 2 ]; then
            echo "," >> "$TMP_CONFIG"
        fi
    else
        # Create new config
        echo "{" > "$TMP_CONFIG"
    fi
    
    # Add new repo entry
    cat >> "$TMP_CONFIG" << EOL
    "$REPO_KEY": {
        "name": "$REPO_NAME",
        "description": "$REPO_DESC",
        "path": "$REPO_PATH",
        "git_url": "$REPO_URL",
        "enabled": true
    }
}
EOL
    
    # Save back to config file
    cat "$TMP_CONFIG" > "$CODE_RAG_CONFIG_PATH"
    rm "$TMP_CONFIG"
    
    echo -e "\n${GREEN}Repository '$REPO_NAME' added successfully!${NC}"
    echo -e "${YELLOW}Building knowledge base to include new repository...${NC}\n"
    
    # Rebuild index
    # Check if parallel indexing script exists
    if [ -f "$SCRIPT_DIR/scripts/parallel_index.py" ]; then
        python scripts/parallel_index.py
    else
        python scripts/index_code.py
    fi
    
    echo -e "\n${GREEN}Didi's knowledge base updated with new repository!${NC}"
}

# List all repositories in the system
function list_repos() {
    print_header
    echo -e "${YELLOW}Repositories in Didi's knowledge base:${NC}\n"
    
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
            echo -e "    Path: $REPO_PATH"
            echo -e "    Status: $STATUS"
            echo
        done
    else
        echo -e "${RED}No repositories configured yet.${NC}"
        echo -e "${YELLOW}Use 'add-repo' command to add repositories.${NC}"
    fi
}

# Enable/disable a repository
function toggle_repo() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Repository key is required${NC}"
        echo -e "Usage: ./didi.sh enable-repo <repo_key>"
        echo -e "   or: ./didi.sh disable-repo <repo_key>"
        exit 1
    fi
    
    print_header
    REPO_KEY="$1"
    MODE="$2"
    
    if [ ! -f "$CODE_RAG_CONFIG_PATH" ]; then
        echo -e "${RED}No repositories configured yet.${NC}"
        exit 1
    fi
    
    # Check if repo exists
    if ! grep -q "\"$REPO_KEY\":" "$CODE_RAG_CONFIG_PATH"; then
        echo -e "${RED}Repository '$REPO_KEY' not found.${NC}"
        echo -e "${YELLOW}Use 'list-repos' to see available repositories.${NC}"
        exit 1
    fi
    
    # Create temporary file
    TMP_CONFIG=$(mktemp)
    
    if [ "$MODE" = "enable" ]; then
        ENABLE="true"
        echo -e "${YELLOW}Enabling repository '$REPO_KEY'...${NC}\n"
    else
        ENABLE="false"
        echo -e "${YELLOW}Disabling repository '$REPO_KEY'...${NC}\n"
    fi
    
    # Replace enabled status
    cat "$CODE_RAG_CONFIG_PATH" | sed "s/\"$REPO_KEY\":\([^{]*{\)/\"$REPO_KEY\":\1/g" | \
        sed "/\"$REPO_KEY\":/,/}/{s/\"enabled\":[[:space:]]*[^,}]*/\"enabled\": $ENABLE/g}" > "$TMP_CONFIG"
    
    # Save back to config file
    cat "$TMP_CONFIG" > "$CODE_RAG_CONFIG_PATH"
    rm "$TMP_CONFIG"
    
    REPO_NAME=$(cat "$CODE_RAG_CONFIG_PATH" | grep -A10 "\"$REPO_KEY\":" | grep "\"name\":" | head -1 | cut -d'"' -f4)
    
    if [ "$MODE" = "enable" ]; then
        echo -e "\n${GREEN}Repository '$REPO_NAME' enabled successfully!${NC}"
    else
        echo -e "\n${GREEN}Repository '$REPO_NAME' disabled successfully!${NC}"
    fi
    
    echo -e "${YELLOW}Rebuilding knowledge base with updated repositories...${NC}\n"
    
    # Rebuild index
    # Check if parallel indexing script exists
    if [ -f "$SCRIPT_DIR/scripts/parallel_index.py" ]; then
        python scripts/parallel_index.py
    else
        python scripts/index_code.py
    fi
    
    echo -e "\n${GREEN}Didi's knowledge base updated!${NC}"
}

# Check status of the system
function check_status() {
    print_header
    echo -e "${YELLOW}Checking Didi's system status...${NC}\n"
    
    # Check persistent storage
    echo -e "${BLUE}Persistent Storage:${NC}"
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
    
    # Check Docker
    echo -e "\n${BLUE}Docker:${NC}"
    if command -v docker &> /dev/null; then
        echo -e "  Docker: ${GREEN}Installed${NC}"
        if command -v docker-compose &> /dev/null; then
            echo -e "  Docker Compose: ${GREEN}Installed${NC}"
        else
            echo -e "  Docker Compose: ${RED}Not installed${NC}"
        fi
    else
        echo -e "  Docker: ${RED}Not installed${NC}"
    fi
    
    # Check Python environment
    echo -e "\n${BLUE}Environment:${NC}"
    echo -e "  Python version: ${GREEN}$(python --version)${NC}"
    
    echo -e "\n${BLUE}Required Python packages:${NC}"
    pip list | grep -E "llama-index|chroma|sentence-transformers|transformers|torch"
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
    docker)
        run_docker
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