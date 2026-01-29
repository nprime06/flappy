#!/bin/bash
# wrapper to submit vae training job
#
# USAGE:
#   ./submit_vae.sh
#   ./submit_vae.sh --run-dir /path/to/run  # resume run (use full absolute path)

set -euo pipefail

# always 1 gpu (no ddp setup)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR=""
NUM_CPUS=16
TOTAL_MEM=128
TIME="6:00:00"

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--run-dir DIR]"
            exit 1
            ;;
    esac
done


# create run directory if not resuming
RUNS_DIR="${SCRIPT_DIR}/runs"
if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${RUNS_DIR}/vae_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

# build train script arguments
TRAIN_ARGS="--run-dir $RUN_DIR"

sbatch \
    --job-name=vae-train \
    --partition=mit_normal_gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${TOTAL_MEM}G \
    --gres=gpu:h200:1 \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,TRAIN_ARGS="$TRAIN_ARGS" \
    "${SCRIPT_DIR}/train_vae.sh"

echo "logs in $RUN_DIR"
