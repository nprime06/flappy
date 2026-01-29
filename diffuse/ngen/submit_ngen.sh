#!/bin/bash
# wrapper to submit ngen training job
#
# USAGE:
#   ./submit_ngen.sh --latent-vod /path/to/dir
#   ./submit_ngen.sh --latent-vod /path/to/dir --gpus 2
#   ./submit_ngen.sh --latent-vod /path/to/dir --run-dir /path/to/run    # resume existing run (use full absolute path)
#   ./submit_ngen.sh --latent-vod /path/to/dir --reflow /path/to/ckpt    # reflow

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR=""
REFLOW=""
LATENT_VOD=""
NUM_GPUS=1
CPUS_PER_GPU=8
MEM_PER_GPU=128
NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
TOTAL_MEM=$((NUM_GPUS * MEM_PER_GPU))
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
        --reflow)
            REFLOW="$2"
            shift 2
            ;;
        --latent-vod)
            LATENT_VOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --latent-vod DIR [--gpus N] [--run-dir DIR] [--reflow CKPT]"
            exit 1
            ;;
    esac
done

if [[ -z "$LATENT_VOD" ]]; then
    echo "Error: --latent-vod is required"
    echo "Usage: $0 --latent-vod DIR [--gpus N] [--run-dir DIR] [--reflow CKPT]"
    exit 1
fi

# create run directory if not resuming
RUNS_DIR="${SCRIPT_DIR}/runs"
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

# build train script arguments
TRAIN_ARGS="--run-dir $RUN_DIR"
if [[ -n "$REFLOW" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --reflow $REFLOW"
fi
TRAIN_ARGS="$TRAIN_ARGS --latent-vod $LATENT_VOD"

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
    "${SCRIPT_DIR}/train_ngen.sh"

echo "logs in $RUN_DIR"
