#!/bin/bash
# set -e

RUNS=$1
START=${2:-1}  # Default start is 1 if not specified

SCRIPT_DIR="/home/jmg/code/python/MML_Suite/MML_Suite/configs/work_chapter_four/msp_improv/baseline_selective/"
echo "Script directory: $SCRIPT_DIR"


# Find all YAML files under the script's directory
CONFIGS=$(find "$SCRIPT_DIR" -type f -name "*.yaml" ! -name "*_template.yaml")
for CONFIG in $CONFIGS; do
    echo "Running experiments for config: $CONFIG"
    
    for run in $(seq "$START" "$RUNS"); do
        python -u MML_Suite/train_federated.py --config "$CONFIG" --run "$run"
        
        # if [ $? -ne 0 ]; then
        #     echo "Experiment failed for config: $CONFIG on run: $run"
        #     exit 1
        # fi
    done
done
