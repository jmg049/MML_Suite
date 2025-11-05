#!/bin/bash
set -e

RUNS=$1
CONFIG="MML_Suite/configs/work_chapter_two/mmimdb/cmams/image_to_text.yaml"
START=${3:-1} # Default value of 1 if the third argument is not provided

for run in $(seq "$START" "$RUNS"); do
    python -u MML_Suite/train_cmam.py --config "$CONFIG" --run "$run"

    # Exit if not successful
    if [ $? -ne 0 ]; then
        echo "Experiment failed.
        Exiting..."
        exit 1
    fi

done

CONFIG="MML_Suite/configs/work_chapter_two/mmimdb/cmams/text_to_image.yaml"
for run in $(seq "$START" "$RUNS"); do
    python -u MML_Suite/train_cmam.py --config "$CONFIG" --run "$run"

    # Exit if not successful
    if [ $? -ne 0 ]; then
        echo "Experiment failed.
        Exiting..."
        exit 1
    fi

done