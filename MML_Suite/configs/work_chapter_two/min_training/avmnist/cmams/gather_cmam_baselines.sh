#!/bin/bash
set -e

RUNS=$1
START=${2:-1}  # Default start = 1 if not provided

# Pattern for all generated configs
CONFIG_DIR="MML_Suite/configs/work_chapter_two/min_training/avmnist/cmams"
CONFIG_PATTERNS=("avmnist_I_A_base_*.yaml")

for pattern in "${CONFIG_PATTERNS[@]}"; do
    for CONFIG in "$CONFIG_DIR"/$pattern; do
        [ -e "$CONFIG" ] || continue  # skip if no files match
        echo "Running config: $CONFIG"

        for run in $(seq "$START" "$RUNS"); do
            echo "  → Run $run"
            python -u MML_Suite/train_cmam.py --config "$CONFIG" --run "$run"

            if [ $? -ne 0 ]; then
                echo "❌ Experiment failed for config $CONFIG (run $run). Exiting..."
                exit 1
            fi
        done
    done
done
