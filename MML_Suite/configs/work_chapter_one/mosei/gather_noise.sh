#!/bin/bash
set -e

RUNS=$1
START=${2:-1} # Default start is 1 if not specified

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for N in $(seq 1 30); do

    # for run in $(seq "$START" "$RUNS"); do
        python -u MML_Suite/train_multimodal.py --config $SCRIPT_DIR/mosei_noise_hyp.yaml --run "$RUNS" --skip-train --append-folder ${N}

        if [ $? -ne 0 ]; then
            echo "Experiment failed on run: $run"
            exit 1
        fi
    # done
done
