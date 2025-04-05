#!/bin/bash
# Didi installation script - makes Didi accessible system-wide

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SYMLINK_DIR="$HOME/.local/bin"

# Make sure the bin directory exists
mkdir -p "$SYMLINK_DIR"

# Create symlinks
echo "Creating symbolic links in $SYMLINK_DIR..."
ln -sf "$SCRIPT_DIR/didi.sh" "$SYMLINK_DIR/didi"
ln -sf "$SCRIPT_DIR/d" "$SYMLINK_DIR/d"

# Check if the bin directory is in PATH
if [[ ":$PATH:" != *":$SYMLINK_DIR:"* ]]; then
    echo "Adding $SYMLINK_DIR to your PATH..."
    
    # Detect shell
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.bashrc"
    fi
    
    echo "export PATH=\"\$PATH:$SYMLINK_DIR\"" >> "$SHELL_RC"
    echo "Added to $SHELL_RC. Please restart your shell or run: source $SHELL_RC"
else
    echo "$SYMLINK_DIR is already in your PATH."
fi

# Add shell completion
echo "Setting up shell completion for Didi..."
COMPLETION_DIR="$HOME/.local/share/bash-completion/completions"
mkdir -p "$COMPLETION_DIR"

cat > "$COMPLETION_DIR/didi" << 'EOF'
_didi_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main commands
    opts="get see g details d ask a status update index help"
    
    # If we're completing a command, show commands
    if [[ ${COMP_CWORD} -eq 1 ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}

complete -F _didi_completion didi d
EOF

echo "
Installation complete! ✨ Didi is now accessible globally.

You can now use these commands from anywhere:
  d get \"query\"           - Find code matches
  d see \"query\"           - Same as get (alias)
  d ask \"question\"        - Ask Didi a question
  
Or even shorter:
  d g \"query\"             - Quick get (alias)
  d a \"question\"          - Ask a question (alias)
  
To load shell aliases for even more shortcuts:
  source $SCRIPT_DIR/aliases.sh
  
To add this automatically to your shell startup:
  echo \"source $SCRIPT_DIR/aliases.sh\" >> $SHELL_RC

" 