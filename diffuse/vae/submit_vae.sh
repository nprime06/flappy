#!/bin/bash
# wrapper to submit vae training job
#
# USAGE:
#   ./submit_vae.sh
#   ./submit_vae.sh --gpus 2
#   ./submit_vae.sh --run-dir /path/to/run  # resume run (use full absolute path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR=""
NUM_GPUS=2
CPUS_PER_GPU=8
MEM_PER_GPU=128
TIME="6:00:00"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpus N] [--run-dir DIR]"
            exit 1
            ;;
    esac
done

NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
TOTAL_MEM=$((NUM_GPUS * MEM_PER_GPU))

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
    --gres=gpu:h200:$NUM_GPUS \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,NUM_GPUS=$NUM_GPUS,TRAIN_ARGS="$TRAIN_ARGS" \
    "${SCRIPT_DIR}/train_vae.sh"

echo "logs in $RUN_DIR"
