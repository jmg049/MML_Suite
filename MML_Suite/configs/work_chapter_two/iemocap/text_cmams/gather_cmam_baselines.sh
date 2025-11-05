#!/bin/bash
set -e

RUNS=$1
START=${2:-1}  # Default start is 1 if not specified

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all YAML files under the script's directory
CONFIGS=$(find "$SCRIPT_DIR" -type f -name "*.yaml" ! -name "*_template.yaml")
for CONFIG in $CONFIGS; do
    
    for run in $(seq "$START" "$RUNS"); do
        echo "Running run: $run for config: $CONFIG"
        python -u MML_Suite/train_cmam.py --config "$CONFIG" --run "$run" --skip-train
        
        if [ $? -ne 0 ]; then
            echo "Experiment failed for config: $CONFIG on run: $run"
            exit 1
        fi
    done
done
