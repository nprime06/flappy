#!/bin/bash
# Wrapper script to submit VAE training job
#
# USAGE:
#   ./submit_ae.sh                           # New run
#   ./submit_ae.sh --run-dir /path/to/run    # Resume existing run (use full absolute path)
#   ./submit_ae.sh --time 12:00:00           # New run with custom time limit

set -euo pipefail

# Defaults
RUN_DIR=""
TIME="6:00:00"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--run-dir DIR] [--time HH:MM:SS]"
            exit 1
            ;;
    esac
done

# Create run directory if not resuming (so logs go there)
RUNS_DIR="/home/willzhao/flappy/diffuse/ae/runs"
if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${RUNS_DIR}/vae_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

# Build train script arguments
TRAIN_ARGS="--run-dir $RUN_DIR"

echo "Submitting VAE training job:"
echo "  CPUs: 8"
echo "  Memory: 64G"
echo "  Time: $TIME"
echo "  Run dir: $RUN_DIR"

# Submit job with dynamic resource allocation
sbatch \
    --job-name=vae-train \
    --partition=mit_normal_gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --gres=gpu:h200:1 \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,TRAIN_ARGS="$TRAIN_ARGS" \
    /home/willzhao/flappy/diffuse/ae/train_ae.sh

echo "Job submitted. Logs will be in: $RUN_DIR"
