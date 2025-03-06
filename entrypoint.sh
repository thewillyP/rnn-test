#!/bin/bash
set -e

# Set the user variable, using the $USER environment variable, or default to 'wlp9800' if not set
USER_NAME="${USER:-wlp9800}"

# Function to append to .bashrc if the line doesn't exist
append_if_not_exists() {
    local line="$1"
    local file="$2"
    
    grep -q "$line" "$file" || echo "$line" >> "$file"
}

# Define the path to the .bashrc
USER_BASHRC="/home/${USER_NAME}/.bashrc"

touch /home/$USER_NAME/.bashrc

# Ensure the necessary environment variables and configurations are set in .bashrc

# 1. Source the Docker-to-Singularity script if not already sourced
append_if_not_exists "source /.singularity.d/env/10-docker2singularity.sh" "$USER_BASHRC"

# 2. Set the LD_LIBRARY_PATH if not already set
append_if_not_exists 'export LD_LIBRARY_PATH="/.singularity.d/libs"' "$USER_BASHRC"

# 3. Apply the changes by sourcing .bashrc
source "$USER_BASHRC"

# Check if the sweep ID argument is provided
if [ -z "$1" ]; then
    echo "Error: No sweep ID provided."
    exit 1
fi

git config --global --add safe.directory /rnn-test

# Run the wandb agent with the provided sweep ID
echo "Starting the wandb agent with sweep ID: $1"
wandb agent --count 1 "$1"
