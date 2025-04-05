#!/bin/bash
# Didi shell aliases
# Add to your shell: source /path/to/didi/aliases.sh

# Get the directory where Didi is installed
DIDI_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Main alias - just 'd' for Didi
alias d="$DIDI_DIR/didi.sh"

# Common command aliases
alias d-get="$DIDI_DIR/didi.sh get"
alias d-see="$DIDI_DIR/didi.sh see"
alias d-g="$DIDI_DIR/didi.sh g"
alias d-ask="$DIDI_DIR/didi.sh ask"
alias d-a="$DIDI_DIR/didi.sh a"
alias d-details="$DIDI_DIR/didi.sh details"
alias d-d="$DIDI_DIR/didi.sh d"
alias d-status="$DIDI_DIR/didi.sh status"
alias d-update="$DIDI_DIR/didi.sh update"

# Add tab completion for the base command
complete -W "get see g details d ask a status update index help" d

echo "Didi aliases loaded! You can now use 'd' as a shortcut."
echo "Examples:"
echo "  d get \"user authentication\""
echo "  d see \"websocket connection\""
echo "  d ask \"How does the contest system work?\"" 