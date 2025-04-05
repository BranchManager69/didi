#!/bin/bash
# d - Quick shortcut for Didi commands
# Usage:
#  d get "query"     - Find code matches
#  d see "query"     - Same as get (alias)
#  d ask "question"  - Ask a question about the code

DIDI_PATH="$(dirname "$(realpath "$0")")"
"$DIDI_PATH/didi.sh" "$@" 