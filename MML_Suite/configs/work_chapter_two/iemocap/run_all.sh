#!/bin/bash

RUNS=$1
START=${2:-1}  # Default start is 1 if not specified
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all "gather_cmam_baselines.sh" files under the script's sub-directories
CONFIGS=$(find "$SCRIPT_DIR" -type f -name "gather_cmam_baselines.sh")
for CONFIG in $CONFIGS; do
    # Extract the directory name from the path
    DIR_NAME=$(dirname "$CONFIG")
    echo "Processing directory: $DIR_NAME"    
    echo "Using config script: $CONFIG"
    bash "$CONFIG" "$RUNS" "$START"
    if [ $? -ne 0 ]; then
        echo "Experiment failed for config: $CONFIG"
    fi
done