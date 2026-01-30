#!/bin/bash
# wrapper to submit RL training job
#
# USAGE:
#   ./submit_main.sh train_ppo.py
#   ./submit_main.sh train_ppo.py --run-dir /path/to/run  # resume run (use full absolute path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE=""
RUN_DIR=""
NUM_CPUS=8
TOTAL_MEM=64
TIME="06:00:00"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        *)
            if [[ -z "$FILE" ]]; then
                FILE="$1"
            else
                echo "Unknown option: $1"
                echo "Usage: $0 FILE [--run-dir DIR]"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$FILE" ]]; then
    echo "Error: FILE is required"
    echo "Usage: $0 FILE [--run-dir DIR]"
    exit 1
fi

RUN_PATH="${SCRIPT_DIR}/${FILE}"
if [[ ! -f "$RUN_PATH" ]]; then
    echo "Error: FILE '$FILE' not found at '$RUN_PATH'" >&2
    exit 1
fi

# Create run directory if not resuming
RUNS_DIR="${SCRIPT_DIR}/runs"
if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${RUNS_DIR}/$(basename "$FILE" .py)_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

# Build train script arguments
TRAIN_ARGS="--run-dir $RUN_DIR"

sbatch \
    --job-name=flappy-bot \
    --partition=mit_normal_gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${TOTAL_MEM}G \
    --gres=gpu:h200:1 \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,FILE="$FILE",TRAIN_ARGS="$TRAIN_ARGS" \
    "${SCRIPT_DIR}/train_main.sh"

echo "logs in $RUN_DIR"
