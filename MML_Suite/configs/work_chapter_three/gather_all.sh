#!/bin/bash
set -e

RUNS=$1
START=${2:-1}  # Default start is 1 if not specified

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/mosei/gather_baselines.sh" "$RUNS" "$START"
bash "$SCRIPT_DIR/iemocap/gather_baselines.sh" "$RUNS" "$START"
bash "$SCRIPT_DIR/msp_improv/gather_baselines.sh" "$RUNS" "$START"
