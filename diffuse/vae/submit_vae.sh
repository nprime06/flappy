#!/bin/bash
# Wrapper script to submit VAE training job with proper GPU allocation
#
# USAGE:
#   ./submit_vae.sh                           # New run
#   ./submit_vae.sh --run-dir /path/to/run    # Resume existing run (use full absolute path)

set -euo pipefail

# Always use 1 GPU (no DDP setup)
RUN_DIR=""

# Parse arguments
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

# Fixed resources for 1 GPU
NUM_CPUS=16
TOTAL_MEM=128

# Create run directory if not resuming (so logs go there)
RUNS_DIR="/home/willzhao/flappy/diffuse/vae/runs"
if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${RUNS_DIR}/vae_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

# Build train script arguments
TRAIN_ARGS="--run-dir $RUN_DIR"

echo "Submitting VAE training job:"
echo "  CPUs: $NUM_CPUS"
echo "  Memory: ${TOTAL_MEM}G"
echo "  Run dir: $RUN_DIR"

# Submit job
sbatch \
    --job-name=vae-train \
    --partition=mit_normal_gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${TOTAL_MEM}G \
    --gres=gpu:h200:1 \
    --time=6:00:00 \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,TRAIN_ARGS="$TRAIN_ARGS" \
    /home/willzhao/flappy/diffuse/vae/train_vae.sh

echo "Job submitted. Logs will be in: $RUN_DIR"
