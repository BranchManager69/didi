#\!/usr/bin/env python3
"""
Model Profile Management Tool for Didi

This script provides commands to list, create, switch and manage different model profiles
for Didi, allowing easy switching between different models and configurations.

Note: This file was renamed from profile.py to model_profile.py to avoid conflicts with Python's built-in profile module.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path to allow importing config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROFILES_DIR, list_profiles, create_profile, get_current_profile, DEFAULT_PROFILES

def display_profiles():
    """Display all available profiles."""
    profiles = list_profiles()
    
    # Get current profile
    current_profile_name = os.environ.get("DIDI_MODEL_PROFILE", "default")
    
    print("\n=== Available Profiles ===")
    for profile_name, display_name in profiles.items():
        current_marker = "* " if profile_name == current_profile_name else "  "
        print(f"{current_marker}{profile_name}: {display_name}")
    
    print("\nCurrent profile: " + current_profile_name)
    
    # Display current profile details
    current_profile = get_current_profile()
    print("\n=== Current Profile Details ===")
    for key, value in current_profile.items():
        print(f"{key}: {value}")

def switch_profile(profile_name):
    """Switch to a different profile."""
    profiles = list_profiles()
    
    if profile_name not in profiles:
        print(f"Error: Profile '{profile_name}' not found.")
        print("Available profiles:")
        for name in profiles.keys():
            print(f"- {name}")
        return False
    
    # Create a shell script to set the environment variable
    script_content = f"""#\!/bin/bash
export DIDI_MODEL_PROFILE="{profile_name}"
echo "Switched to profile: {profile_name}"
"""
    
    # Write to a temporary file
    script_path = Path(PROFILES_DIR) / "switch_profile.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"Profile switched to '{profile_name}'")
    print(f"To activate, run: source {script_path}")
    return True

def create_new_profile(args):
    """Create a new profile based on command line arguments."""
    profile_name = args.name
    
    # Check if profile already exists
    if (PROFILES_DIR / f"{profile_name}.json").exists():
        if not args.force:
            print(f"Error: Profile '{profile_name}' already exists. Use --force to overwrite.")
            return False
    
    # Start with a copy of the default profile
    base_profile = "default"
    if args.base:
        base_profile = args.base
    
    # Check if base profile exists
    base_profile_path = PROFILES_DIR / f"{base_profile}.json"
    if not base_profile_path.exists():
        print(f"Error: Base profile '{base_profile}' not found.")
        return False
    
    # Load base profile
    with open(base_profile_path, "r") as f:
        profile_data = json.load(f)
    
    # Update with provided values
    if args.display_name:
        profile_data["name"] = args.display_name
    
    if args.llm_model:
        profile_data["llm_model"] = args.llm_model
    
    if args.embed_model:
        profile_data["embed_model"] = args.embed_model
    
    if args.collection_name:
        profile_data["collection_name"] = args.collection_name
    
    if args.chunk_size:
        profile_data["chunk_size"] = args.chunk_size
    
    if args.chunk_overlap:
        profile_data["chunk_overlap"] = args.chunk_overlap
    
    if args.context_window:
        profile_data["context_window"] = args.context_window
    
    if args.max_new_tokens:
        profile_data["max_new_tokens"] = args.max_new_tokens
    
    if args.temperature:
        profile_data["temperature"] = args.temperature
    
    # Create the profile
    success = create_profile(profile_name, profile_data)
    if success:
        print(f"Created new profile '{profile_name}'")
        print("Profile settings:")
        for key, value in profile_data.items():
            print(f"  {key}: {value}")
    else:
        print(f"Failed to create profile '{profile_name}'")
    
    return success

def reset_profiles():
    """Reset all profiles to their default values."""
    for profile_name, profile_data in DEFAULT_PROFILES.items():
        create_profile(profile_name, profile_data)
    print("All default profiles have been reset to their original values.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Manage Didi model profiles")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List profiles command
    list_parser = subparsers.add_parser("list", help="List available profiles")
    
    # Switch profile command
    switch_parser = subparsers.add_parser("switch", help="Switch to a different profile")
    switch_parser.add_argument("profile", help="Name of the profile to switch to")
    
    # Create new profile command
    create_parser = subparsers.add_parser("create", help="Create a new profile")
    create_parser.add_argument("name", help="Name of the new profile")
    create_parser.add_argument("--force", action="store_true", help="Overwrite existing profile")
    create_parser.add_argument("--base", help="Base profile to extend")
    create_parser.add_argument("--display-name", help="Display name for the profile")
    create_parser.add_argument("--llm-model", help="LLM model to use")
    create_parser.add_argument("--embed-model", help="Embedding model to use")
    create_parser.add_argument("--collection-name", help="Collection name in ChromaDB")
    create_parser.add_argument("--chunk-size", type=int, help="Chunk size for text splitting")
    create_parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap for text splitting")
    create_parser.add_argument("--context-window", type=int, help="Context window size")
    create_parser.add_argument("--max-new-tokens", type=int, help="Maximum new tokens to generate")
    create_parser.add_argument("--temperature", type=float, help="Temperature for generation")
    
    # Reset profiles command
    reset_parser = subparsers.add_parser("reset", help="Reset all profiles to defaults")
    
    args = parser.parse_args()
    
    # Process commands
    if args.command == "list" or not args.command:
        display_profiles()
    elif args.command == "switch":
        switch_profile(args.profile)
    elif args.command == "create":
        create_new_profile(args)
    elif args.command == "reset":
        reset_profiles()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
