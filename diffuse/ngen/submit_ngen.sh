#!/bin/bash
# Wrapper script to submit NGEN training job with proper GPU allocation
#
# USAGE:
#   ./submit_ngen.sh                           # New run, 1 GPU
#   ./submit_ngen.sh --gpus 4                  # New run, 4 GPUs
#   ./submit_ngen.sh --run-dir /path/to/run    # Resume existing run (use full absolute path)
#   ./submit_ngen.sh --reflow /path/to/ckpt    # Reflow training
#   ./submit_ngen.sh --gpus 4 --run-dir /path  # Resume with 4 GPUs (use full absolute path)

set -euo pipefail

# Defaults
NUM_GPUS=1
RUN_DIR=""
REFLOW=""
TIME="6:00:01"

# Parse arguments
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
        --reflow)
            REFLOW="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpus N] [--run-dir DIR] [--reflow CKPT] [--time HH:MM:SS]"
            exit 1
            ;;
    esac
done

# Calculate resources based on GPU count
CPUS_PER_GPU=8
NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
MEM_PER_GPU=128
TOTAL_MEM=$((NUM_GPUS * MEM_PER_GPU))

# Create run directory if not resuming (so logs go there)
RUNS_DIR="/home/willzhao/flappy/diffuse/ngen/runs"
if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    if [[ -n "$REFLOW" ]]; then
        PREFIX="reflow"
    else
        PREFIX="ngen"
    fi
    RUN_DIR="${RUNS_DIR}/${PREFIX}_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

# Build train script arguments
TRAIN_ARGS="--run-dir $RUN_DIR"
if [[ -n "$REFLOW" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --reflow $REFLOW"
fi

echo "Submitting NGEN training job:"
echo "  GPUs: $NUM_GPUS"
echo "  CPUs: $NUM_CPUS"
echo "  Memory: ${TOTAL_MEM}G"
echo "  Time: $TIME"
echo "  Run dir: $RUN_DIR"
if [[ -n "$REFLOW" ]]; then
    echo "  Reflow: $REFLOW"
fi

# Submit job with dynamic resource allocation
sbatch \
    --job-name=ngen-train \
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
    /home/willzhao/flappy/diffuse/ngen/train_ngen.sh

echo "Job submitted. Logs will be in: $RUN_DIR"
